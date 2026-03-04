import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,

)
from sklearn.calibration import CalibrationDisplay


def plot_roc_comparison(models: dict, X_test, y_test, save_path: str):
    """
    models: dict like {"logistic": model1, "svm_rbf": model2, "xgb": model3}
    """
    _ensure_dir(save_path)

    plt.figure()
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)

    plt.title("ROC Comparison")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _ensure_dir(save_path: str):
    d = os.path.dirname(save_path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_roc(model, X_test, y_test, save_path):
    _ensure_dir(save_path)
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_pr_curve(model, X_test, y_test, save_path):
    _ensure_dir(save_path)
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(model, X_test, y_test, save_path, n_bins: int = 10):
    """
    Reliability diagram: predicted probability vs observed frequency.
    """
    _ensure_dir(save_path)
    CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=n_bins)
    plt.title("Calibration Curve")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, save_path):
    _ensure_dir(save_path)
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix (default threshold)")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_at_threshold(y_true, y_proba, threshold: float, save_path):
    """
    Confusion matrix using a custom threshold on predicted probabilities.
    """
    _ensure_dir(save_path)
    y_pred = (np.asarray(y_proba) >= float(threshold)).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f"Confusion Matrix (threshold = {threshold:.3f})")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_shap_summary_xgb(model, X, save_path, max_display: int = 20):
    """
    Robust SHAP summary plot using shap.Explainer on predict_proba.
    Avoids XGBoost TreeExplainer base_score parsing issues.
    """
    import shap
    import pandas as pd
    import numpy as np

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # If model is a Pipeline, use it directly (it has predict_proba)
    # background sample for speed
    X_bg = X.sample(n=min(len(X), 200), random_state=42)

    # shap.Explainer wants a callable; we explain prob(class=1)
    def f(data):
        # data arrives as numpy sometimes; convert back to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=X.columns)
        return model.predict_proba(data)[:, 1]

    explainer = shap.Explainer(f, X_bg)
    shap_values = explainer(X_bg)

    # Bar summary plot
    import matplotlib.pyplot as plt
    plt.figure()
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance (XGBoost via shap.Explainer)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_shap_dependence_xgb(model, X, feature_name: str, save_path):
    """
    Robust SHAP dependence plot using shap.Explainer.
    """
    import shap
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X_bg = X.sample(n=min(len(X), 300), random_state=42)

    def f(data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=X.columns)
        return model.predict_proba(data)[:, 1]

    explainer = shap.Explainer(f, X_bg)
    shap_values = explainer(X_bg)

    plt.figure()
    shap.plots.scatter(shap_values[:, feature_name], show=False)
    plt.title(f"SHAP Dependence: {feature_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_permutation_importance(
    model,
    X,
    y,
    save_csv_path: str,
    save_fig_path: str,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "f1",
    top_n: int = 20,
):
    """
    Computes permutation importance on (X, y), saves:
      1) CSV (feature, importance_mean, importance_std)
      2) bar plot for top_n features
    X should be a pandas DataFrame to keep feature names.
    """
    _ensure_dir(save_csv_path)
    _ensure_dir(save_fig_path)

    if not hasattr(X, "columns"):
        raise ValueError("X must be a pandas DataFrame so we can keep feature names.")

    r = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )

    df_imp = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)

    df_imp.to_csv(save_csv_path, index=False)

    df_top = df_imp.head(top_n).iloc[::-1]  # reverse for horizontal bar plot

    plt.figure(figsize=(8, max(4, 0.3 * len(df_top))))
    plt.barh(df_top["feature"], df_top["importance_mean"])
    plt.title(f"Permutation Importance (top {top_n})")
    plt.xlabel(f"Mean Δ {scoring} (higher = more important)")
    plt.tight_layout()
    plt.savefig(save_fig_path, bbox_inches="tight")
    plt.close()

    return df_imp