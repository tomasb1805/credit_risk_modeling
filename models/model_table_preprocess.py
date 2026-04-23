# Python libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# Machine learning + eval libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, auc, roc_curve

# Configuration
load_dotenv()
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

TABLE = os.getenv("MODEL_TABLE_PATH")

def load_model_table(table_path: str = TABLE) -> pd.DataFrame:
    """Load the database table to be preprocessed."""
    return pd.read_csv(table_path)

def refine_mainstream(row: pd.Series) -> str:
    """
    Refinement of the 'mainstream' persona.
    Separates it into sub-groups using existing features.
    """
    if row['income'] >= 40_000 and row['dti_ratio'] <= 0.50:
        return 'mainstream_stable'

    if row['income'] < 40_000 and row['dti_ratio'] <= 0.40:
        return 'mainstream_low_income'

    if row['dti_ratio'] > 0.50:
        return 'mainstream_leveraged'

    return 'mainstream_other'

def assign_mainstream_refine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the 'mainstream' persona refinement to the DataFrame.
    """
    df_refined = df.copy()
    mask = df['persona_name'] == 'mainstream'

    df_refined.loc[mask, 'persona_name'] = df_refined[mask].apply(refine_mainstream, axis=1)

    return df_refined

def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    For mainstream rows, behavioral features are NaN.
    Add binary flags so the model knows the absence is informative.
    """
    behavioral_cols = [
        'avg_monthly_inflow', 'income_stability_ratio',
        'spend_to_income_ratio', 'months_net_negative_6m',
        'cash_withdrawal_ratio'
    ]
    for col in behavioral_cols:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    return df

def preprocess_df(df: pd.DataFrame):
    """Preprocess and partition DataFrame."""
    df_model = df.copy()
    cols = ['persona_name', 'loan_id', 'years_employed', 'loan_term_years', 'default', 'dti_ratio_pct']

    X = df_model.drop(columns=cols)
    y = df_model['default']

    numerical_features = ['age', 'income', 'loan_amount', 'credit_score', 'months_employed',
                          'num_credit_lines', 'interest_rate', 'loan_term', 'dti_ratio',
                          'has_mortgage', 'has_dependents', 'has_cosigner', 'avg_monthly_inflow',
                          'income_stability_ratio', 'spend_to_income_ratio', 
                          # 'months_net_negative_6m',
                          'age', 'income', 'debt_to_income', 'rate_per_term', 'credit_utilisation',
                          'employment_stability'
                          'cash_withdrawal_ratio',
                        #   'avg_monthly_inflow_missing', 'income_stability_ratio_missing',
                        #   'spend_to_income_ratio_missing', 'months_net_negative_6m_missing',
                        #   'cash_withdrawal_ratio_missing'
                        ]

    categorical_features = ['education', 'employment_type', 'marital_status', 'loan_purpose']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=13)

    preprocessing = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    return X_train, X_test, y_train, y_test, preprocessing

def build_pipeline(X_train, y_train, preprocessing):
    """Constructs the classification pipeline without fitting."""
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    pipeline = Pipeline([          
        ('preprocessing', preprocessing),
        ('model', XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=11
        ))
    ])

    return pipeline

def hyper_tuning(pipeline, X_train, y_train):
    """Executes hyperparameter optimization via RandomizedSearchCV."""
    param_grid = {
        'model__n_estimators':     [100, 200, 400],
        'model__max_depth':        [3, 4, 5, 6],
        'model__learning_rate':    [0.01, 0.05, 0.1],
        'model__subsample':        [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
        'model__min_child_weight': [1, 3, 5],
        'model__reg_alpha':        [0, 0.1, 1.0],
        'model__reg_lambda':       [1.0, 2.0, 5.0]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=50,
        scoring='roc_auc',
        cv=5,
        random_state=11,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print(f"Cross-validation Best AUC: {search.best_score_:.4f}")
    print(f"Optimal Parameters: {search.best_params_}")

    return search.best_estimator_

def evaluate_model(fitted_pipeline, X_test, y_test):
    """Computes test set ROC AUC metrics using the optimized estimator."""
    y_pred_proba = fitted_pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Out-of-sample Test ROC-AUC: {test_auc:.4f}")

    return y_test, y_pred_proba

def plot_roc_auc(y_test, y_pred_proba):
    """Generate ROC AUC visualization."""
    plt.figure(figsize=(7, 5))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for XGBoost Model')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    df = load_model_table()
    df = assign_mainstream_refine(df)
    df = add_missing_flags(df)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_df(df)

    # Instantiate un-fitted pipeline
    model_pipeline = build_pipeline(X_train, y_train, preprocessor)
    
    # Execute cross-validation tuning
    tuned_pipeline = hyper_tuning(model_pipeline, X_train, y_train)
    
    # Evaluate optimized estimator on holdout set
    y_true, y_probs = evaluate_model(tuned_pipeline, X_test, y_test)
    
    # Render diagnostics
    plot_roc_auc(y_true, y_probs)
