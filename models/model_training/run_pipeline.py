from build_model_dataset import (
    load_model_table,
    assign_mainstream_refine,
    add_missing_flags,
    feature_engineering,
    preprocess_df
)

from train_xgb_model  import build_pipeline, hyper_tuning
from evaluate_model   import evaluate_model, plot_roc_auc
from explain_model    import (
    build_shap_explainer,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_dependence,
    plot_example_applicants
)
import numpy as np


if __name__ == "__main__":

    # === STEP 1: Prepare data ===
    df = load_model_table()
    df = assign_mainstream_refine(df)
    df = add_missing_flags(df)
    df = feature_engineering(df)

    # preprocess_df returns splits + preprocessor — all stored in variables
    X_train, X_test, y_train, y_test, preprocessor, df_test = preprocess_df(df)

    # === STEP 2: Train ===
    # Pass the outputs of step 1 directly into the training functions
    model_pipeline = build_pipeline(y_train, preprocessor)
    tuned_pipeline = hyper_tuning(model_pipeline, X_train, y_train)

    # === STEP 3: Evaluate ===
    # tuned_pipeline from step 2 flows into evaluate_model
    y_true, y_proba = evaluate_model(tuned_pipeline, X_test, y_test)
    plot_roc_auc(y_true, y_proba)

    # === STEP 4: Explain ===
    # Both tuned_pipeline (step 2) and X_train/X_test (step 1) flow in here
    explainer   = build_shap_explainer(tuned_pipeline, X_train)

    sample_idx  = np.random.default_rng(11).choice(len(X_test), 2000, replace=False)
    X_sample    = X_test.iloc[sample_idx]
    shap_values = compute_shap_values(explainer, tuned_pipeline, X_sample)
    y_proba_sample = y_proba[sample_idx]

    plot_shap_summary(shap_values)
    plot_shap_dependence(shap_values, tuned_pipeline, X_sample, top_n=5)
    plot_example_applicants(shap_values, tuned_pipeline, X_sample, y_proba_sample)
