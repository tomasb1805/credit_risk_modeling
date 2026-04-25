# Python libraries
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Machine learning + eval libraries
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# configuration
load_dotenv()
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

TABLE = os.getenv("MODEL_TABLE_PATH")

def load_model_table(table_path: str = TABLE) -> pd.DataFrame:
    """
    Load the database table to be preprocessed.
    """
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
    Assign the 'mainstream' persona refinement to the DataFrame relevant rows.
    """
    df_refined = df.copy()
    mask = df['persona_name'] == 'mainstream'

    df_refined.loc[mask, 'persona_name'] = df_refined[mask].apply(refine_mainstream, axis=1)

    return df_refined

def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    For mainstream rows, the behavioral features are NaN.
    Add the "_missing_" flag so the model knows understands the absence is intended.
    """
    behavioral_cols = [
        'avg_monthly_inflow', 'income_stability_ratio',
        'spend_to_income_ratio', 'months_net_negative_6m',
        'cash_withdrawal_ratio'
    ]

    for col in behavioral_cols:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer interaction features to improve AUC scoring.
    """
    df['debt_to_income']       = df['loan_amount'] / df['income']
    df['rate_per_term']        = df['interest_rate'] / df['loan_term']
    df['credit_utilisation']   = df['loan_amount'] / df['credit_score']
    df['employment_stability'] = np.log1p(df['income'] * df['months_employed'])

    return df


def preprocess_df(df: pd.DataFrame):
    """
    Preprocess the DataFrame for model instatiation.
    """
    df_model = df.copy()
    cols = ['persona_name', 'loan_id', 'years_employed', 'loan_term_years', 'default', 'dti_ratio_pct']

    X = df_model.drop(columns=cols)
    y = df_model['default']

    numerical_features = ['age', 'income', 'loan_amount', 'credit_score', 'months_employed',
                          'num_credit_lines', 'interest_rate', 'loan_term', 'dti_ratio',
                          'has_mortgage', 'has_dependents', 'has_cosigner', 'avg_monthly_inflow',
                          'income_stability_ratio', 'spend_to_income_ratio',
                          'debt_to_income', 'rate_per_term', 'credit_utilisation',
                          'employment_stability', 'cash_withdrawal_ratio'

                          
                        #   'months_net_negative_6m',
                        #   'avg_monthly_inflow_missing', 'income_stability_ratio_missing',
                        #   'spend_to_income_ratio_missing', 'months_net_negative_6m_missing',
                        #   'cash_withdrawal_ratio_missing'
                        ]

    categorical_features = ['education', 'employment_type', 'marital_status', 'loan_purpose']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=13)

    df_test = df_model.loc[X_test.index]


    preprocessing = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessing, df_test
if __name__ == "__main__":
    df = load_model_table()
    df = assign_mainstream_refine(df)
    df = add_missing_flags(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_df(df)

