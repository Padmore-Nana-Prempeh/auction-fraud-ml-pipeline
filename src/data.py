import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop obvious ID-like columns if present (we’ll refine once we inspect columns)
    drop_candidates = {"Record_ID", "Auction_ID", "Bidder_ID", "id", "ID"}
    cols_to_drop = [c for c in df.columns if c in drop_candidates]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    return df