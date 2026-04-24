import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ---------------------------------------------------------------------------
# Underwriting policy
# ---------------------------------------------------------------------------

APPROVE_THRESHOLD = 0.20   # bad rate ~2.2% — within loss tolerance
REFER_THRESHOLD   = 0.35   # bad rate ~3.9% — borderline, human review

def underwriting_decision(probability: float) -> str:
    """Maps a default probability to an underwriting action."""
    if probability < APPROVE_THRESHOLD:
        return 'APPROVE'
    elif probability <= REFER_THRESHOLD:
        return 'REFER — Manual Review'
    else:
        return 'DECLINE'


# ---------------------------------------------------------------------------
# Helpers: extract readable feature names from the pipeline
# ---------------------------------------------------------------------------

def get_feature_names(pipeline) -> list:
    """
    Extracts transformed column names from the ColumnTransformer.
    Strips sklearn prefixes (num__, cat__) for clean SHAP labels.
    """
    raw_names = (
        pipeline.named_steps['preprocessing']
        .get_feature_names_out()
    )
    # Remove 'num__' and 'cat__' prefixes added by ColumnTransformer
    clean = [n.replace('num__', '').replace('cat__', '') for n in raw_names]
    return clean


def get_shap_input(pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Transforms X through the preprocessor only, returning a numpy array
    that matches the feature space the model was trained on.
    """
    return pipeline.named_steps['preprocessing'].transform(X)


# ---------------------------------------------------------------------------
# SHAP explainer setup
# ---------------------------------------------------------------------------

def build_shap_explainer(pipeline, X_train: pd.DataFrame):
    """
    Builds a TreeExplainer using the fitted model extracted from the pipeline.
    Uses a background sample of 500 rows for efficiency on 51k dataset.
    """
    model        = pipeline.named_steps['model']
    X_train_t    = get_shap_input(pipeline, X_train)
    background   = shap.sample(X_train_t, 500, random_state=11)

    explainer    = shap.TreeExplainer(model, background)
    return explainer


def compute_shap_values(explainer, pipeline, X: pd.DataFrame):
    """
    Computes SHAP Explanation object with clean feature names attached.
    Returns an shap.Explanation object ready for all plot types.
    """
    X_t           = get_shap_input(pipeline, X)
    feature_names = get_feature_names(pipeline)

    shap_values = explainer(X_t)
    shap_values.feature_names = feature_names   # attach clean names
    return shap_values


# ---------------------------------------------------------------------------
# Plot 1 — SHAP Summary (Beeswarm)
# ---------------------------------------------------------------------------

def plot_shap_summary(shap_values, max_display: int = 20):
    """
    Global feature importance beeswarm plot.
    Each dot is one applicant; colour = feature value (red=high, blue=low).
    Ranked by mean |SHAP| — shows which features drive default risk most.
    """
    plt.figure()
    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Summary — Global Feature Impact on Default Probability')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2 — Dependence Plots for Top Features
# ---------------------------------------------------------------------------

def plot_shap_dependence(shap_values, pipeline, X_test: pd.DataFrame,
                         top_n: int = 5):
    """
    Dependence plots for the top_n most important features.
    Shows how each feature's value relates to its SHAP contribution,
    coloured by the feature that most interacts with it (auto-selected).
    """
    X_t           = get_shap_input(pipeline, X_test)
    feature_names = get_feature_names(pipeline)
    shap_matrix   = shap_values.values

    # Rank features by mean |SHAP|
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    top_indices   = np.argsort(mean_abs_shap)[::-1][:top_n]

    for idx in top_indices:
        feat_name = feature_names[idx]
        plt.figure(figsize=(7, 4))
        shap.dependence_plot(
            ind=idx,
            shap_values=shap_matrix,
            features=X_t,
            feature_names=feature_names,
            interaction_index='auto',   # auto-selects best interaction feature
            show=False
        )
        plt.title(f'SHAP Dependence — {feat_name}')
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Plot 3 — Waterfall Plot for Individual Applicants
# ---------------------------------------------------------------------------

def plot_waterfall(shap_values, pipeline, X_test: pd.DataFrame,
                   y_proba: np.ndarray, applicant_idx: int):
    """
    Waterfall plot for a single applicant.
    Shows exactly why the model gave this specific probability,
    with underwriting decision banner at the top.
    """
    prob     = y_proba[applicant_idx]
    decision = underwriting_decision(prob)

    decision_colours = {
        'APPROVE':              'green',
        'REFER — Manual Review': 'orange',
        'DECLINE':              'red'
    }
    colour = decision_colours[decision]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)

    shap.plots.waterfall(
        shap_values[applicant_idx],
        max_display=12,
        show=False
    )

    plt.title(
        f'Applicant #{applicant_idx} — Default Probability: {prob:.1%}  |  '
        f'Decision: {decision}',
        fontsize=11,
        color=colour,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


def plot_example_applicants(shap_values, pipeline, X_test: pd.DataFrame,
                            y_proba: np.ndarray):
    """
    Plots waterfall explanations for three representative applicants:
    one from each underwriting band (Approve / Refer / Decline).
    """
    approve_idx = np.where(y_proba < APPROVE_THRESHOLD)[0][0]
    refer_idx   = np.where(
        (y_proba >= APPROVE_THRESHOLD) & (y_proba <= REFER_THRESHOLD)
    )[0][0]
    decline_idx = np.where(y_proba > REFER_THRESHOLD)[0][0]

    for idx, label in [
        (approve_idx, 'Auto-Approve Example'),
        (refer_idx,   'Refer / Manual Review Example'),
        (decline_idx, 'Auto-Decline Example')
    ]:
        print(f"\n--- {label} ---")
        plot_waterfall(shap_values, pipeline, X_test, y_proba, idx)


# ---------------------------------------------------------------------------
# Entry point — add to __main__ after evaluate_model()
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- After fitting tuned_pipeline and splitting data ---

    y_proba = tuned_pipeline.predict_proba(X_test)[:, 1]

    # 1. Build explainer (uses training data as background distribution)
    explainer   = build_shap_explainer(tuned_pipeline, X_train)

    # 2. Compute SHAP values for the test set
    #    Use a sample of 2000 for speed — full 10k test set is slow
    sample_idx  = np.random.default_rng(11).choice(len(X_test), 2000, replace=False)
    X_sample    = X_test.iloc[sample_idx]
    shap_values = compute_shap_values(explainer, tuned_pipeline, X_sample)
    y_proba_sample = y_proba[sample_idx]

    # 3. Global summary — which features matter most overall
    plot_shap_summary(shap_values, max_display=20)

    # 4. Dependence plots — how top features drive risk
    plot_shap_dependence(shap_values, tuned_pipeline, X_sample, top_n=5)

    # 5. Waterfall — one example per underwriting band
    plot_example_applicants(shap_values, tuned_pipeline, X_sample, y_proba_sample)
