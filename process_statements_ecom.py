import pandas as pd
from extract_statement_info import extract_statement_info, _is_null

# Maps each statement column -> (KPI display name, statement_type)
STATEMENT_COLUMNS = {
    "Purchase_Rate_3_Month_Window_Statement":       ("Purchase Rate",       "rolling"),
    "Avg_Session_Revenue_3_Month_Window_Statement": ("Avg Session Revenue", "rolling"),
    "Avg_Discount_3_Month_Window_Statement":        ("Avg Discount",        "rolling"),
}

OUTPUT_COLS = [
    "Segment", "KPI", "Statement Type",
    "Direction", "Change (%/$)",
    "Prior Period", "Prior Value",
    "Current Period", "Current Value",
]


def process_csv(input_path: str, output_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    records = []

    for _, row in df.iterrows():
        segment = row["Segment"]

        for col, (kpi_name, stmt_type) in STATEMENT_COLUMNS.items():
            if col not in df.columns:
                continue

            raw = str(row[col]).strip()
            display_type = "Rolling 3-Month" if stmt_type == "rolling" else "Month-over-Month"

            if _is_null(raw):
                records.append({
                    "Segment":        segment,
                    "KPI":            kpi_name,
                    "Statement Type": display_type,
                    "Direction":      "No Anomaly Detected",
                    "Change (%/$)":   None,
                    "Prior Period":   None,
                    "Prior Value":    None,
                    "Current Period": None,
                    "Current Value":  None,
                })
                continue

            info = extract_statement_info(raw, statement_type=stmt_type)

            if info is None:
                records.append({
                    "Segment":        segment,
                    "KPI":            kpi_name,
                    "Statement Type": display_type,
                    "Direction":      "No Anomaly Detected",
                    "Change (%/$)":   None,
                    "Prior Period":   None,
                    "Prior Value":    None,
                    "Current Period": None,
                    "Current Value":  None,
                })
            else:
                records.append({
                    "Segment":        segment,
                    "KPI":            kpi_name,
                    "Statement Type": display_type,
                    "Direction":      info.get("Direction"),
                    "Change (%/$)":   info.get("Change"),
                    "Prior Period":   info.get("Prior Period"),
                    "Prior Value":    info.get("Prior Value"),
                    "Current Period": info.get("Current Period"),
                    "Current Value":  info.get("Current Value"),
                })

    result_df = pd.DataFrame(records, columns=OUTPUT_COLS)

    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")

    return result_df


if __name__ == "__main__":
    INPUT  = r"C:\Users\naopk00\OneDrive - Chubb\Desktop\Chubb\porfolio+\ecom_anomalies.csv"
    OUTPUT = r"C:\Users\naopk00\OneDrive - Chubb\Desktop\Chubb\porfolio+\ecom_statements_parsed.csv"

    result = process_csv(INPUT, OUTPUT)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 50)
    print(result.to_string(index=False))
