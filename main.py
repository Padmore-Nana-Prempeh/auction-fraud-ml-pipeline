import os
from src.config import Config
from src.data import load_data, basic_clean
from src.split import split_train_val_test
from src.train import tune_and_train_all

def main():
    cfg = Config()

    df = basic_clean(load_data(cfg.data_path))

    # If your target column name differs, we’ll fix it after the first run prints columns
    if cfg.target_col not in df.columns:
        print("Columns found:\n", list(df.columns))
        raise ValueError(f"Target column '{cfg.target_col}' not found. Update Config.target_col.")

    # Ensure target is 0/1
    y = df[cfg.target_col]
    if set(y.unique()) == {-1, 1}:
        df[cfg.target_col] = (df[cfg.target_col] == 1).astype(int)
    elif set(y.unique()) == {0, 1}:
        pass
    else:
        # try to coerce common formats
        df[cfg.target_col] = y.astype(int)

    splits = split_train_val_test(
        df=df,
        target=cfg.target_col,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state
    )

    os.makedirs(cfg.reports_dir, exist_ok=True)
    results = tune_and_train_all(splits, reports_dir=cfg.reports_dir)

    print("\nDone. Results saved to reports/metrics.json")
    print(results)

if __name__ == "__main__":
    main()