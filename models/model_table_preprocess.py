import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, cross_val_score
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

def preprocess_df(df: pd.DataFrame):
    """Preprocess and partition DataFrame."""
    df_model = df[~(df['persona_name'] == "mainstream")]
    cols = ['persona_name', 'loan_id', 'years_employed', 'loan_term_years', 'default', 'dti_ratio_pct']

    X = df_model.drop(columns=cols)
    y = df_model['default']

    numerical_features = ['age', 'income', 'loan_amount', 'credit_score', 'months_employed',
                          'num_credit_lines', 'interest_rate', 'loan_term', 'dti_ratio',
                          'has_mortgage', 'has_dependents', 'has_cosigner', 'avg_monthly_inflow',
                          'income_stability_ratio', 'spend_to_income_ratio',
                          'months_net_negative_6m', 'cash_withdrawal_ratio']

    categorical_features = ['education', 'employment_type', 'marital_status', 'loan_purpose']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

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

def build_and_train_pipeline(X_train, y_train, preprocessing):
    """Instantiate and fit the model pipeline, accounting for class imbalance."""
    
    # Dynamic calculation of scale_pos_weight
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('model', XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=11
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    """Evaluate model utilizing cross-validation and test set ROC AUC."""
    
    scores = cross_val_score(
        pipeline, 
        X_train, 
        y_train, 
        cv=5, 
        scoring='roc_auc', 
        error_score="raise"
    )

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Test ROC-AUC: {test_auc:.4f}")
    print(f"Cross-validation ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

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
    X_train, X_test, y_train, y_test, preprocessor = preprocess_df(df)
    model_pipeline = build_and_train_pipeline(X_train, y_train, preprocessor)
    y_true, y_probs = evaluate_model(model_pipeline, X_train, y_train, X_test, y_test)
    plot_roc_auc(y_true, y_probs)
