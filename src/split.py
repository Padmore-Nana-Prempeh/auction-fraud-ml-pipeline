from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

def split_train_val_test(df: pd.DataFrame, target: str, test_size: float, val_size: float, random_state: int) -> SplitData:
    y = df[target]
    X = df.drop(columns=[target])

    # test split first
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # val split from the remainder
    val_frac_of_tmp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac_of_tmp, random_state=random_state, stratify=y_tmp
    )

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)