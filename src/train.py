import os
import pandas as pd
from sklearn.model_selection import ParameterGrid

from .metrics import evaluate, save_json
from .models import make_logistic_baseline, make_svm_rbf, make_xgb_classifier
from .plots import (
    plot_roc,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_calibration_curve,
    plot_confusion_matrix_at_threshold,
    plot_permutation_importance,
    plot_shap_summary_xgb,
    plot_shap_dependence_xgb,
    plot_roc_comparison,
)
from .thresholds import find_best_threshold
from .persist import save_model, save_json as save_json_simple


def fit_predict_proba(model, X_train, y_train, X_eval):
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1]


def _finalize_and_log(
    model_key: str,
    model,
    best_params: dict,
    splits,
    X_tv,
    y_tv,
    fig_dir: str,
    model_dir: str,
    results: dict,
):
    # Fit on train+val
    model.fit(X_tv, y_tv)

    # Predict probabilities
    p_test = model.predict_proba(splits.X_test)[:, 1]

    # default threshold results
    test_default = evaluate(splits.y_test, p_test)

    # choose threshold based on VAL (already computed earlier typically)
    # here we recompute threshold using TV's VAL part would not exist; so instead:
    # we store threshold when selecting best model. That comes from `results[model_key]["threshold"]`.
    # (we’ll set it outside for each model)
    thr = results[model_key]["threshold"]

    # plot ROC / PR / calibration
    # plot ROC / PR / calibration (your plots.py expects estimator-based calls)
    plot_roc(
        model, splits.X_test, splits.y_test,
        os.path.join(fig_dir, f"roc_{model_key}.png")
    )
    plot_pr_curve(
        model, splits.X_test, splits.y_test,
        os.path.join(fig_dir, f"pr_{model_key}.png")
    )
    plot_calibration_curve(
        model, splits.X_test, splits.y_test,
        os.path.join(fig_dir, f"cal_{model_key}.png")
    )

    # confusion matrices: default + tuned threshold
    plot_confusion_matrix(
        model, splits.X_test, splits.y_test,
        os.path.join(fig_dir, f"cm_{model_key}_default.png")
    )
    plot_confusion_matrix_at_threshold(
        splits.y_test, p_test, thr,
        os.path.join(fig_dir, f"cm_{model_key}_thr.png")
    )

# permutation importance: save CSV + figure
    plot_permutation_importance(
        model=model,
        X=splits.X_test,
        y=splits.y_test,
        save_csv_path=os.path.join(fig_dir, f"permutation_importance_{model_key}.csv"),
        save_fig_path=os.path.join(fig_dir, f"permutation_importance_{model_key}.png"),
        scoring="f1",
    )

    # save trained model
    model_path = os.path.join(model_dir, f"best_{model_key}.joblib")
    save_model(model, model_path)

    # save compact model card 
    model_card = {
        "model": model_key,
        "best_params": best_params,
        "best_threshold": thr,
        "test_default_threshold_metrics": test_default.__dict__,
        "artifacts": {
            "model_path": model_path,
            "roc_curve": f"figures/roc_{model_key}.png",
            "pr_curve": f"figures/pr_{model_key}.png",
            "calibration_curve": f"figures/cal_{model_key}.png",
            "cm_default": f"figures/cm_{model_key}_default.png",
            "cm_thresholded": f"figures/cm_{model_key}_thr.png",
            "permutation_importance_csv": f"figures/permutation_importance_{model_key}.csv",
            "permutation_importance_plot": f"figures/permutation_importance_{model_key}.png",
        }
    }
    save_json_simple(model_card, os.path.join(model_dir, f"model_card_{model_key}.json"))


def tune_and_train_all(splits, reports_dir: str):
    os.makedirs(reports_dir, exist_ok=True)
    fig_dir = os.path.join(reports_dir, "figures")
    model_dir = os.path.join(reports_dir, "models")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    results = {}

    # Keep feature names by using pandas concat
    X_tv = pd.concat([splits.X_train, splits.X_val], axis=0)
    y_tv = pd.concat([splits.y_train, splits.y_val], axis=0)

    # -----------------------------
    # 1) Logistic tuning
    # -----------------------------
    logit_grid = {"C": [0.1, 0.3, 1.0, 3.0, 10.0]}
    best = None

    for params in ParameterGrid(logit_grid):
        m = make_logistic_baseline(**params)
        p_val = fit_predict_proba(m.model, splits.X_train, splits.y_train, splits.X_val)
        r = evaluate(splits.y_val, p_val)

        thr = find_best_threshold(splits.y_val, p_val, metric="f1")
        if best is None or r.f1 > best["val"].f1:
            best = {"params": params, "model": m.model, "val": r, "thr": thr}

    results["logistic"] = {
        "best_params": best["params"],
        "val": best["val"].__dict__,
        "threshold": float(best["thr"]),
    }
    _finalize_and_log("logistic", best["model"], best["params"], splits, X_tv, y_tv, fig_dir, model_dir, results)
    best_logit_model = best["model"]
    # -----------------------------
    # 2) SVM RBF tuning
    # -----------------------------
    svm_grid = {"C": [0.3, 1.0, 3.0, 10.0], "gamma": ["scale", "auto"]}
    best = None

    for params in ParameterGrid(svm_grid):
        m = make_svm_rbf(**params)
        p_val = fit_predict_proba(m.model, splits.X_train, splits.y_train, splits.X_val)
        r = evaluate(splits.y_val, p_val)

        thr = find_best_threshold(splits.y_val, p_val, metric="f1")
        if best is None or r.f1 > best["val"].f1:
            best = {"params": params, "model": m.model, "val": r, "thr": thr}

    results["svm_rbf"] = {
        "best_params": best["params"],
        "val": best["val"].__dict__,
        "threshold": float(best["thr"]),
    }
    _finalize_and_log("svm_rbf", best["model"], best["params"], splits, X_tv, y_tv, fig_dir, model_dir, results)
    best_svm_model = best["model"]
    # -----------------------------
    # 3) XGBoost tuning + SHAP
    # -----------------------------
    xgb_grid = {
        "n_estimators": [300, 500],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1.0, 2.0],
    }

    best = None
    for params in ParameterGrid(xgb_grid):
        m = make_xgb_classifier(**params)
        p_val = fit_predict_proba(m.model, splits.X_train, splits.y_train, splits.X_val)
        r = evaluate(splits.y_val, p_val)

        thr = find_best_threshold(splits.y_val, p_val, metric="f1")
        if best is None or r.f1 > best["val"].f1:
            best = {"params": params, "model": m.model, "val": r, "thr": thr}

    results["xgb"] = {
        "best_params": best["params"],
        "val": best["val"].__dict__,
        "threshold": float(best["thr"]),
    }

    # fit + plots + saving
    _finalize_and_log("xgb", best["model"], best["params"], splits, X_tv, y_tv, fig_dir, model_dir, results)
    best_xgb_model = best["model"]
    # SHAP artifacts (XGB only)
    # Use X_test for explainability so it reflects generalization
    try:
        plot_shap_summary_xgb(
            best["model"],
            splits.X_test,
            os.path.join(fig_dir, "shap_summary_xgb.png"),
        )

    # Pick a top feature for dependence (simple example)
        first_feature = splits.X_test.columns[0]

        plot_shap_dependence_xgb(
            best["model"],
            splits.X_test,
            first_feature,
            os.path.join(fig_dir, "shap_dep_xgb.png"),
        )

    except Exception as e:
        print(f"[WARN] SHAP plots failed: {e}")
    

    # Add SHAP artifacts to results
    results["xgb"]["artifacts"] = results["xgb"].get("artifacts", {})
    results["xgb"]["artifacts"].update({
        "shap_summary": "figures/shap_summary_xgb.png",
        "shap_dependence_example": "figures/shap_dep_xgb.png",
    })



    # -----------------------------
    # ROC comparison across models
    # -----------------------------
    plot_roc_comparison(
        {
            "Logistic": best_logit_model,
            "SVM (RBF)": best_svm_model,
            "XGBoost": best_xgb_model,
        },
        splits.X_test,
        splits.y_test,
        os.path.join(fig_dir, "roc_comparison.png"),
    )

    # -----------------------------
    # Save global metrics.json
    # -----------------------------
    save_json(results, os.path.join(reports_dir, "metrics.json"))
    return results