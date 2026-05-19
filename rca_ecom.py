"""
Root Cause Analysis: ecom_anomalies.csv  ->  ecom_sessions.csv
==============================================================
For each anomalous segment in ecom_anomalies.csv this script:
  1. Filters ecom_sessions.csv to the segment + time windows.
  2. Computes each KPI broken down by DEPT_LINE_DESC and by CHANNEL.
  3. Compares current window (Sep 2024 - Nov 2024) vs prior (Aug 2024 - Oct 2024).
  4. Ranks sub-segments by their contribution to the KPI shift.

GLM layer:
  For KPIs with a direct row-level column (purchase_rate, avg_session_revenue,
  avg_discount) the script fits:  target ~ C(period) * C(sub_segment)
  Predicted marginal means (response scale) are diffed to produce glm_delta,
  which replaces raw delta as the ranking key for those KPIs.
  revenue_share and traffic_share are derived ratio metrics with no row-level
  equivalent; they retain the original raw-delta ranking.

GLM families:
  purchase_rate       -> Binomial  / logit   (PURCHASED_FLAG  0/1)
  avg_session_revenue -> Gamma     / log     (SESSION_REVENUE  >0 for purchases)
  avg_discount        -> Gaussian  / identity(DISCOUNT_PCT     continuous)

Output:
  reports/rca_contributions.csv   - ranked sub-segment contributions
  reports/rca_summary.csv         - one-liner per segment x KPI
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from process_statements_ecom import process_csv, STATEMENT_COLUMNS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE         = Path(__file__).parent
ANOMALY_PATH = BASE / "ecom_anomalies.csv"
RAW_PATH     = BASE / "ecom_sessions.csv"
OUT_DIR      = BASE / "reports_project"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# KPI name mapping: statement display name -> internal name used in RCA
# ---------------------------------------------------------------------------
STMT_KPI_TO_INTERNAL: dict[str, str] = {
    "Purchase Rate":        "purchase_rate",
    "Avg Session Revenue":  "avg_session_revenue",
    "Avg Discount":         "avg_discount",
}

TOP_N = 5   # sub-segments to report per dimension


# ---------------------------------------------------------------------------
# Helpers: parse period strings produced by extract_statement_info
# ---------------------------------------------------------------------------

def _parse_period_range(period_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    "Aug 2024 - Oct 2024"  ->  (Timestamp("2024-08-01"), Timestamp("2024-10-31"))
    "Sep 2024 - Nov 2024"  ->  (Timestamp("2024-09-01"), Timestamp("2024-11-30"))
    """
    left, right = [p.strip() for p in period_str.split(" - ")]
    start = pd.Timestamp(datetime.strptime(left,  "%b %Y"))
    end   = pd.Timestamp(datetime.strptime(right, "%b %Y")) + pd.offsets.MonthEnd(0)
    return start, end


def _build_segment_info(extracted: pd.DataFrame) -> dict[str, dict]:
    """
    Build a per-segment lookup from the extracted statements DataFrame.

    Returns
    -------
    {
      segment: {
        "current_start": Timestamp,
        "current_end":   Timestamp,
        "prior_start":   Timestamp,
        "prior_end":     Timestamp,
        "anomalous_kpis": [internal_kpi, ...]
      }
    }

    Only rolling 3-month rows are used (Statement Type == "Rolling 3-Month").
    Segments with no non-null statement fall back to globally derived windows.
    """
    info: dict[str, dict] = {}

    for _, row in extracted[extracted["Statement Type"] == "Rolling 3-Month"].iterrows():
        seg = row["Segment"]
        if seg not in info:
            info[seg] = {
                "current_start": None, "current_end": None,
                "prior_start":   None, "prior_end":   None,
                "anomalous_kpis": [],
            }

        kpi_internal = STMT_KPI_TO_INTERNAL.get(row["KPI"])
        if row["Direction"] == "No Anomaly Detected" or kpi_internal is None:
            continue

        # Derive time windows from the first available statement for this segment
        if info[seg]["current_start"] is None and pd.notna(row["Current Period"]):
            info[seg]["current_start"], info[seg]["current_end"] = _parse_period_range(row["Current Period"])
            info[seg]["prior_start"],   info[seg]["prior_end"]   = _parse_period_range(row["Prior Period"])

        if kpi_internal not in info[seg]["anomalous_kpis"]:
            info[seg]["anomalous_kpis"].append(kpi_internal)

    return info


# ---------------------------------------------------------------------------
# Segment -> filter mapping
# ---------------------------------------------------------------------------
DEPT_COL_MAP = {
    "DEPT":      "DEPT_DESC",
    "DEPT_SUB":  "DEPT_SUB_DESC",
    "DEPT_LINE": "DEPT_LINE_DESC",
}


def parse_segment(segment: str) -> tuple[str, str]:
    """Return (filter_column, filter_value) for a segment name."""
    for prefix, col in DEPT_COL_MAP.items():
        if segment.startswith(prefix + " "):
            _, val = segment.split(" ", 1)
            return col, val
    if segment.startswith("CHANNEL "):
        return "CHANNEL", segment[8:]
    raise ValueError(f"Cannot parse segment: {segment!r}")


# ---------------------------------------------------------------------------
# KPI computation helpers
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Aggregate session-level data to KPI values per group_col.

    KPIs returned:
      purchase_rate       - sum(PURCHASED_FLAG) / session count
      avg_session_revenue - mean(SESSION_REVENUE)  [revenue per session]
      avg_discount        - mean(DISCOUNT_PCT)
      sessions            - count of records
      revenue             - sum(SESSION_REVENUE)   [kept for mix KPI denominator]
    """
    grp = df.groupby(group_col, dropna=False)
    agg = pd.DataFrame({
        "sessions":       grp.size(),
        "purchases":      grp["PURCHASED_FLAG"].sum(),
        "revenue":        grp["SESSION_REVENUE"].sum(),
        "discount_sum":   grp["DISCOUNT_PCT"].sum(),
        "discount_count": grp["DISCOUNT_PCT"].count(),
    })
    agg["purchase_rate"]       = agg["purchases"] / agg["sessions"]
    agg["avg_session_revenue"] = agg["revenue"]   / agg["sessions"]
    agg["avg_discount"]        = agg["discount_sum"] / agg["discount_count"]
    return agg.drop(columns=["purchases", "discount_sum", "discount_count"])



# ---------------------------------------------------------------------------
# GLM configuration and fitting
# ---------------------------------------------------------------------------

# Maps internal KPI name -> (raw CSV column, statsmodels family)
KPI_GLM_CONFIG: dict[str, tuple[str, object]] = {
    "purchase_rate":       ("PURCHASED_FLAG",   sm.families.Binomial()),
    "avg_session_revenue": ("SESSION_REVENUE",  sm.families.Gamma(link=sm.families.links.Log())),
    "avg_discount":        ("DISCOUNT_PCT",      sm.families.Gaussian()),
}


def fit_glm_contributions(
    df_current: pd.DataFrame,
    df_prior: pd.DataFrame,
    group_col: str,
    kpi: str,
) -> pd.DataFrame:
    """
    Fit:  target ~ C(period) * C(sub_segment)
    on stacked current (period=1) + prior (period=0) row-level data.

    Family is chosen per KPI:
      purchase_rate       -> Binomial / logit
      avg_session_revenue -> Gamma    / log
      avg_discount        -> Gaussian / identity

    Returns a DataFrame indexed by sub-segment value with columns:
      glm_pred_prior    - model-predicted KPI in prior period (response scale)
      glm_pred_current  - model-predicted KPI in current period (response scale)
      glm_delta         - glm_pred_current minus glm_pred_prior
      glm_pvalue        - p-value of the period effect for that sub-segment

    Returns an empty DataFrame when GLM is not applicable or fails.
    """
    config = KPI_GLM_CONFIG.get(kpi)
    if config is None:
        return pd.DataFrame()

    target_col, family = config

    def _prep(df: pd.DataFrame, period_val: int) -> pd.DataFrame:
        sub = df[[target_col, group_col]].copy()
        sub["period"] = period_val
        return sub

    stacked = pd.concat([_prep(df_current, 1), _prep(df_prior, 0)], ignore_index=True)
    stacked = stacked.dropna(subset=[target_col, group_col])

    # Gamma log-link requires strictly positive values
    # For SESSION_REVENUE this filters out non-purchased sessions (revenue=0)
    if isinstance(family, sm.families.Gamma):
        stacked = stacked[stacked[target_col] > 0]

    # Rename to safe patsy identifiers before building formula
    stacked = stacked.rename(columns={group_col: "sub_seg", target_col: "target"})

    formula = "target ~ C(period) * C(sub_seg)"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.glm(formula, data=stacked, family=family).fit(disp=False)
    except Exception:
        return pd.DataFrame()

    # Predicted marginal means for each sub-segment x period combination
    sub_segs  = stacked["sub_seg"].unique()
    pred_grid = pd.DataFrame({
        "sub_seg": np.tile(sub_segs, 2),
        "period":  np.repeat([0, 1], len(sub_segs)),
    })
    pred_grid["predicted"] = model.predict(pred_grid)

    pivot = pred_grid.pivot(index="sub_seg", columns="period", values="predicted")
    pivot.columns = ["glm_pred_prior", "glm_pred_current"]
    pivot["glm_delta"] = pivot["glm_pred_current"] - pivot["glm_pred_prior"]

    # p-value for the period effect per sub-segment:
    #   reference category -> main period term   C(period)[T.1]
    #   all others         -> interaction term   C(period)[T.1]:C(sub_seg)[T.<value>]
    pvals   = model.pvalues
    ref_cat = sorted(stacked["sub_seg"].unique())[0]

    glm_pval: dict = {}
    for sg in sub_segs:
        key = (
            "C(period)[T.1]"
            if sg == ref_cat
            else f"C(period)[T.1]:C(sub_seg)[T.{sg}]"
        )
        glm_pval[sg] = pvals.get(key, np.nan)

    pivot["glm_pvalue"] = pd.Series(glm_pval)
    pivot.index.name = group_col   # restore original column name

    return pivot


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe(val, digits: int = 6):
    """Round a scalar to `digits` places; return None if NaN/None."""
    try:
        return round(float(val), digits) if pd.notna(val) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Load & pre-process raw data
# ---------------------------------------------------------------------------
print("Loading raw data ...")
raw = pd.read_csv(RAW_PATH)
raw["SESSION_DATE"] = pd.to_datetime(raw["SESSION_DATE"])

# All-portfolio totals for mix denominators (per period)
def period_totals(df: pd.DataFrame) -> dict:
    return {
        "sessions": len(df),
        "revenue":  df["SESSION_REVENUE"].sum(),
    }


# ---------------------------------------------------------------------------
# Load anomaly file and extract statement info
# ---------------------------------------------------------------------------
anomaly_df = pd.read_csv(ANOMALY_PATH)

print("Extracting statement info from anomaly file ...")
extracted_statements = process_csv(ANOMALY_PATH)
segment_info         = _build_segment_info(extracted_statements)

# Derive global windows from the extracted data (used for portfolio-level totals
# and as fallback for segments with no statement). All rows share the same
# reference month, so the first non-null entry is representative.
_window_row = extracted_statements[
    (extracted_statements["Statement Type"] == "Rolling 3-Month") &
    (extracted_statements["Direction"] != "No Anomaly Detected") &
    extracted_statements["Current Period"].notna()
].iloc[0]

CURRENT_START, CURRENT_END = _parse_period_range(_window_row["Current Period"])
PRIOR_START,   PRIOR_END   = _parse_period_range(_window_row["Prior Period"])

raw_current = raw[(raw["SESSION_DATE"] >= CURRENT_START) & (raw["SESSION_DATE"] <= CURRENT_END)]
raw_prior   = raw[(raw["SESSION_DATE"] >= PRIOR_START)   & (raw["SESSION_DATE"] <= PRIOR_END)]

totals_current = period_totals(raw_current)
totals_prior   = period_totals(raw_prior)

print(f"  Current window ({CURRENT_START.date()} - {CURRENT_END.date()}): "
      f"{totals_current['sessions']:,} sessions")
print(f"  Prior window   ({PRIOR_START.date()} - {PRIOR_END.date()}): "
      f"{totals_prior['sessions']:,} sessions")


# ---------------------------------------------------------------------------
# Main RCA loop
# ---------------------------------------------------------------------------
SUB_DIMS = ["DEPT_LINE", "CHANNEL"]   # two breakdown dimensions

DEPT_LINE_COL = "DEPT_LINE_DESC"
CHANNEL_COL   = "CHANNEL"

DIM_COL = {
    "DEPT_LINE": DEPT_LINE_COL,
    "CHANNEL":   CHANNEL_COL,
}

all_contributions: list[dict] = []
all_summaries:     list[dict] = []

for _, row in anomaly_df.iterrows():
    segment      = row["Segment"]
    seg_meta     = segment_info.get(segment, {})
    anomaly_kpis = seg_meta.get("anomalous_kpis", [])

    if not anomaly_kpis:
        print(f"  [SKIP] No anomalous KPIs detected for: {segment!r}")
        continue

    try:
        filter_col, filter_val = parse_segment(segment)
    except ValueError as e:
        print(f"  [SKIP] {e}")
        continue

    # Use segment-specific windows if available, else fall back to global windows
    seg_cur_start = seg_meta.get("current_start") or CURRENT_START
    seg_cur_end   = seg_meta.get("current_end")   or CURRENT_END
    seg_pri_start = seg_meta.get("prior_start")   or PRIOR_START
    seg_pri_end   = seg_meta.get("prior_end")     or PRIOR_END

    seg_raw_cur = raw[(raw["SESSION_DATE"] >= seg_cur_start) & (raw["SESSION_DATE"] <= seg_cur_end)]
    seg_raw_pri = raw[(raw["SESSION_DATE"] >= seg_pri_start) & (raw["SESSION_DATE"] <= seg_pri_end)]

    # Filter to this segment
    seg_current = seg_raw_cur[seg_raw_cur[filter_col] == filter_val]
    seg_prior   = seg_raw_pri[seg_raw_pri[filter_col] == filter_val]

    if seg_current.empty and seg_prior.empty:
        print(f"  [WARN] No data for segment: {segment!r}")
        continue

    print(f"\n{'='*60}")
    print(f"Segment: {segment}  |  anomalous KPIs: {anomaly_kpis}")
    print(f"  Current sessions: {len(seg_current):,}   Prior sessions: {len(seg_prior):,}")

    for dim_name in SUB_DIMS:
        dim_col = DIM_COL[dim_name]

        # Skip DEPT_LINE breakdown if segment is already at product line level
        if filter_col == DEPT_LINE_COL and dim_name == "DEPT_LINE":
            continue
        # Skip CHANNEL breakdown if segment is already a single channel
        if filter_col == CHANNEL_COL and dim_name == "CHANNEL":
            continue

        # Compute KPIs per sub-segment in each period
        agg_cur = compute_kpis(seg_current, dim_col) if not seg_current.empty else pd.DataFrame()
        agg_pri = compute_kpis(seg_prior,   dim_col) if not seg_prior.empty   else pd.DataFrame()


        # Merge current and prior
        merged = agg_cur.add_suffix("_cur").join(
                     agg_pri.add_suffix("_pri"), how="outer"
                 )
        merged.index.name = dim_col

        for kpi in anomaly_kpis:
            cur_col = f"{kpi}_cur"
            pri_col = f"{kpi}_pri"
            if cur_col not in merged.columns or pri_col not in merged.columns:
                continue

            merged[f"{kpi}_delta"]   = merged[cur_col] - merged[pri_col]
            merged[f"{kpi}_pct_chg"] = (
                (merged[cur_col] - merged[pri_col]) / merged[pri_col].abs()
            ).replace([np.inf, -np.inf], np.nan)

            # ------------------------------------------------------------------
            # GLM: target ~ C(period) * C(sub_segment)
            # Fit on row-level stacked data; extract predicted marginal means.
            # glm_delta (response scale) is the ranking key where available;
            # raw kpi_delta is the fallback for mix metrics.
            # ------------------------------------------------------------------
            glm_res = fit_glm_contributions(seg_current, seg_prior, dim_col, kpi)

            if not glm_res.empty:
                merged_glm = merged.join(
                    glm_res[["glm_delta", "glm_pred_current", "glm_pred_prior", "glm_pvalue"]],
                    how="left",
                )
                rank_col = "glm_delta"
            else:
                merged_glm = merged.copy()
                merged_glm["glm_delta"]        = np.nan
                merged_glm["glm_pred_current"] = np.nan
                merged_glm["glm_pred_prior"]   = np.nan
                merged_glm["glm_pvalue"]       = np.nan
                rank_col = f"{kpi}_delta"

            # Rank by absolute GLM delta (or raw delta for mix metrics)
            ranked = (
                merged_glm[[
                    f"{kpi}_delta", cur_col, pri_col, f"{kpi}_pct_chg",
                    "sessions_cur", "sessions_pri",
                    "glm_delta", "glm_pred_current", "glm_pred_prior", "glm_pvalue",
                ]]
                .dropna(subset=[f"{kpi}_delta"])
                .sort_values(rank_col, key=abs, ascending=False)
                .head(TOP_N)
                .reset_index()
            )

            # Compute parent-level KPI values for reference
            if kpi == "purchase_rate":
                parent_cur = seg_current["PURCHASED_FLAG"].mean()
                parent_pri = seg_prior["PURCHASED_FLAG"].mean()
            elif kpi == "avg_session_revenue":
                parent_cur = seg_current["SESSION_REVENUE"].mean()
                parent_pri = seg_prior["SESSION_REVENUE"].mean()
            elif kpi == "avg_discount":
                parent_cur = seg_current["DISCOUNT_PCT"].mean()
                parent_pri = seg_prior["DISCOUNT_PCT"].mean()
            else:
                parent_cur = parent_pri = np.nan

            parent_delta = (
                parent_cur - parent_pri
                if (not np.isnan(parent_cur) and not np.isnan(parent_pri))
                else np.nan
            )

            for rank_pos, r in ranked.iterrows():
                sub_seg_val = r[dim_col]
                all_contributions.append({
                    "segment":              segment,
                    "anomalous_kpi":        kpi,
                    "dimension":            dim_name,
                    "sub_segment":          sub_seg_val,
                    "rank":                 rank_pos + 1,
                    "kpi_current":          _safe(r[cur_col]),
                    "kpi_prior":            _safe(r[pri_col]),
                    "kpi_delta":            _safe(r[f"{kpi}_delta"]),
                    "kpi_pct_change":       _safe(r[f"{kpi}_pct_chg"] * 100, 2) if pd.notna(r[f"{kpi}_pct_chg"]) else None,
                    "sessions_current":     int(r["sessions_cur"]) if pd.notna(r["sessions_cur"]) else None,
                    "sessions_prior":       int(r["sessions_pri"]) if pd.notna(r["sessions_pri"]) else None,
                    "parent_kpi_current":   _safe(parent_cur),
                    "parent_kpi_prior":     _safe(parent_pri),
                    "parent_kpi_delta":     _safe(parent_delta),
                    # GLM columns — None for mix metrics where GLM is not applied
                    "glm_pred_current":     _safe(r["glm_pred_current"]),
                    "glm_pred_prior":       _safe(r["glm_pred_prior"]),
                    "glm_delta":            _safe(r["glm_delta"]),
                    "glm_pvalue":           _safe(r["glm_pvalue"], 4),
                })

            # Summary: top contributor per dimension x KPI
            if not ranked.empty:
                top = ranked.iloc[0]
                all_summaries.append({
                    "segment":          segment,
                    "anomalous_kpi":    kpi,
                    "dimension":        dim_name,
                    "top_contributor":  top[dim_col],
                    "top_kpi_delta":    round(top[f"{kpi}_delta"], 6),
                    "top_pct_change":   round(top[f"{kpi}_pct_chg"] * 100, 2) if not np.isnan(top[f"{kpi}_pct_chg"]) else None,
                    "parent_kpi_delta": round(parent_delta, 6) if parent_delta and not np.isnan(parent_delta) else None,
                    "direction":        "Increased" if (parent_delta and parent_delta > 0) else "Decreased",
                })

            print(f"  [{kpi.upper():22s}] by {dim_name}: top contributor = "
                  f"{ranked.iloc[0][dim_col] if not ranked.empty else 'n/a'!r}")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
contrib_df = pd.DataFrame(all_contributions)
summary_df = pd.DataFrame(all_summaries)

contrib_path = OUT_DIR / "rca_contributions_ecom.csv"
summary_path = OUT_DIR / "rca_summary_ecom.csv"

contrib_df.to_csv(contrib_path, index=False)
summary_df.to_csv(summary_path, index=False)

print(f"\n{'='*60}")
print(f"Saved {len(contrib_df)} contribution rows -> {contrib_path}")
print(f"Saved {len(summary_df)} summary rows     -> {summary_path}")
print("\nDone.")
