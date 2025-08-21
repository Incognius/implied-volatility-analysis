import re
from datetime import datetime
from pathlib import Path
import pandas as pd

def _safe_float(x):
    if pd.isna(x):
        return None
    s = str(x).strip().replace(",", "")
    if s in {"--", "-", ""}:
        return None
    try:
        return float(s)
    except ValueError:
        return None

def parse_expiry_from_filename(path_str: str):
    m = re.search(r"_(\d{4}-\d{2}-\d{2})_option_chain", Path(path_str).name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()

def load_option_chain_long(file_path: str, spot: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    required = ["Strike", "Call LTP", "Put LTP"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found. Available: {list(df.columns)}")

    df["Strike"]   = df["Strike"].apply(_safe_float)
    df["Call LTP"] = df["Call LTP"].apply(_safe_float)
    df["Put LTP"]  = df["Put LTP"].apply(_safe_float)

    calls = df[["Strike", "Call LTP"]].rename(columns={"Strike": "strike", "Call LTP": "ltp"})
    calls["type"] = "call"

    puts  = df[["Strike", "Put LTP"]].rename(columns={"Strike": "strike", "Put LTP": "ltp"})
    puts["type"]  = "put"

    long_df = pd.concat([calls, puts], ignore_index=True)

    expiry_date = parse_expiry_from_filename(file_path)
    long_df["expiry"] = pd.to_datetime(expiry_date) if expiry_date else pd.NaT
    long_df["spot"]   = _safe_float(spot) if spot is not None else None

    long_df = long_df.dropna(subset=["strike", "ltp"]).sort_values(["strike", "type"]).reset_index(drop=True)
    return long_df
