import pandas as pd

KEY_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
]

def load_transactions(file_path):
    # Read as strings first to preserve leading zeros in bank codes.
    df = pd.read_csv(file_path, dtype=str)

    df.columns = df.columns.str.strip()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    for col in ["Amount Received", "Amount Paid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def build_transaction_index(df):
    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in transactions data: {', '.join(missing)}")

    return df.set_index(KEY_COLUMNS, drop=False)
