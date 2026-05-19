import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

MONTHS_PATTERN = (
    r"(January|February|March|April|May|June|"
    r"July|August|September|October|November|December)"
)

NULL_VALUES = {"null", "none", "nan", "", "-"}


def _get_period_label(end_date: datetime, window_months: int) -> str:
    start_date = end_date - relativedelta(months=window_months - 1)
    return f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"


def _is_null(value: str) -> bool:
    return str(value).strip().lower() in NULL_VALUES


def extract_statement_info(statement: str, statement_type: str = None) -> dict | None:
    """
    Extracts structured info from a narrative KPI statement.

    Parameters
    ----------
    statement      : The narrative text.
    statement_type : 'rolling' or 'monthly'. If None, auto-detected.

    Returns None if the statement is null/empty.
    Returns a dict with keys: Direction, Change, Prior Value,
    Current Value, Prior Period, Current Period.
    """
    if _is_null(statement):
        return None

    statement = statement.strip()

    # --- Auto-detect type if not provided ---
    if statement_type is None:
        statement_type = (
            "rolling"
            if re.search(r"rolling\s+\d+\s*month|prior\s+\d+\s*month", statement, re.IGNORECASE)
            else "monthly"
        )

    result = {}

    # --- Direction and change magnitude ---
    change_match = re.search(
        r"(increased|decreased|rose|fell|dropped|grew)\s+by\s+\$?\s*([\d,]+\.?\d*)\s*%?",
        statement, re.IGNORECASE,
    )
    if change_match:
        result["Direction"] = change_match.group(1).capitalize()
        result["Change"]    = change_match.group(2).replace(",", "")

    # --- Prior and current values ---
    # Handles: "from 6.11% to 5.89%", "from $7302.6 ... to $7928.6", "from 7,136.81 to 7,447.52"
    val_match = re.search(
        r"from\s+\$?\s*([\d,]+\.?\d*)\s*%?.*?to\s+\$?\s*([\d,]+\.?\d*)\s*%?",
        statement, re.IGNORECASE | re.DOTALL,
    )
    if val_match:
        result["Prior Value"]   = val_match.group(1).replace(",", "")
        result["Current Value"] = val_match.group(2).replace(",", "")

    # --- Periods ---
    if statement_type == "rolling":
        # Reference month is the single Month YYYY in the statement
        month_match = re.search(rf"{MONTHS_PATTERN}\s+(\d{{4}})", statement, re.IGNORECASE)
        if month_match:
            ref_date = datetime.strptime(
                f"{month_match.group(1)} {month_match.group(2)}", "%B %Y"
            )
            window_match = re.search(r"(\d+)\s*month", statement, re.IGNORECASE)
            window = int(window_match.group(1)) if window_match else 3

            result["Current Period"] = _get_period_label(ref_date, window)
            result["Prior Period"]   = _get_period_label(ref_date - relativedelta(months=1), window)

    else:  # monthly
        # Two explicit months: "from X in Month1 Year1 to Y in Month2 Year2"
        months_found = re.findall(
            rf"{MONTHS_PATTERN}\s+(\d{{4}})", statement, re.IGNORECASE
        )
        if len(months_found) >= 2:
            prior_date   = datetime.strptime(f"{months_found[0][0]} {months_found[0][1]}", "%B %Y")
            current_date = datetime.strptime(f"{months_found[1][0]} {months_found[1][1]}", "%B %Y")
            result["Prior Period"]   = prior_date.strftime("%b %Y")
            result["Current Period"] = current_date.strftime("%b %Y")
        elif len(months_found) == 1:
            current_date = datetime.strptime(f"{months_found[0][0]} {months_found[0][1]}", "%B %Y")
            result["Current Period"] = current_date.strftime("%b %Y")
            result["Prior Period"]   = (current_date - relativedelta(months=1)).strftime("%b %Y")

    return result
