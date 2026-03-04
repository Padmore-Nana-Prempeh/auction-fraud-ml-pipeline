from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    data_path: str = "data/raw/shill_bidding.csv"
    target_col: str = "Class"   # we’ll confirm after we peek at columns
    test_size: float = 0.20
    val_size: float = 0.20      # from remaining after test split
    random_state: int = 42

    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"