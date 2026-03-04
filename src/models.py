import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

@dataclass
class ModelOutput:
    name: str
    model: object

def make_logistic_baseline(C: float = 1.0) -> ModelOutput:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
    ])
    return ModelOutput(name="logistic_baseline", model=pipe)

def make_svm_rbf(C: float = 1.0, gamma: str = "scale") -> ModelOutput:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=C, kernel="rbf", gamma=gamma, probability=True, class_weight="balanced"))
    ])
    return ModelOutput(name="svm_rbf", model=pipe)

def make_xgb_classifier(
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_child_weight: float = 1.0,
) -> ModelOutput:
    # note: we do NOT StandardScale for tree models
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    return ModelOutput(name="xgb", model=clf)