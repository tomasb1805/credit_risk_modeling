import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# example of underwriting policy

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



# extract readable feature names from the pipeline

def get_feature_names(pipeline) -> list:
    """
    Extracts transformed column names from the ColumnTransformer.
    Strips sklearn prefixes (num__, cat__) for clean SHAP labels.
    """
    raw_names = (
        pipeline.named_steps['preprocessing']
        .get_feature_names_out()
    )

    clean = [n.replace('num__', '').replace('cat__', '') for n in raw_names]
    return clean


def get_shap_input(pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Transforms X through the preprocessor only, returning a numpy array
    that matches the feature space the model was trained on.
    """
    return pipeline.named_steps['preprocessing'].transform(X)



# SHAP setup

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



# SHAP Beeswarm Summary 

def plot_shap_summary(shap_values, max_display: int = 20):
    """
    Global feature importance beeswarm plot.
    Each dot is one applicant; colour = feature value (red=high, blue=low).
    Ranked by mean |SHAP| — shows which features drive default risk most.
    """
    fig = plt.figure()
    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Summary — Global Feature Impact on Default Probability')
    plt.tight_layout()
    plt.show()
    plt.close(fig)



# Top Features - plot for individual contributions to model  

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


    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    top_indices   = np.argsort(mean_abs_shap)[::-1][:top_n]

    for idx in top_indices:
        feat_name = feature_names[idx]
        fig = plt.figure(figsize=(7, 4))
        shap.dependence_plot(
            ind=idx,
            shap_values=shap_matrix,
            features=X_t,
            feature_names=feature_names,
            interaction_index='auto',
            show=False
        )
        plt.title(f'SHAP Dependence — {feat_name}')
        plt.tight_layout()
        plt.show()
        plt.close(fig)



# Waterfall Plot for individual applicants
# explains how their features contributed to the 
# final underwriting outcome/decision

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
    plt.close(fig)


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


if __name__ == "__main__":

    y_proba = tuned_pipeline.predict_proba(X_test)[:, 1]

    
    explainer   = build_shap_explainer(tuned_pipeline, X_train)

    # compute SHAP values for the test set.
    # using a sample of 2000 for speed
    sample_idx  = np.random.default_rng(11).choice(len(X_test), 2000, replace=False)
    X_sample    = X_test.iloc[sample_idx]
    shap_values = compute_shap_values(explainer, tuned_pipeline, X_sample)
    y_proba_sample = y_proba[sample_idx]

    plot_shap_summary(shap_values, max_display=20)

    plot_shap_dependence(shap_values, tuned_pipeline, X_sample, top_n=5)

    plot_example_applicants(shap_values, tuned_pipeline, X_sample, y_proba_sample)
