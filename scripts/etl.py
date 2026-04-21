import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# main config settings
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

_ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

INPUT_PATH  = os.getenv("INPUT_PATH")
DB_USER     = os.getenv("ADMIN")
DB_PASSWORD = os.getenv("PASSWORD")
DB_HOST     = os.getenv("HOST", "localhost")
DB_PORT     = os.getenv("PORT", "5432")
DB_NAME     = os.getenv("DB_NAME")




# configuration 
COLUMN_RENAME_MAP = {
    'loanid': 'loan_id',
    'loanamount': 'loan_amount',
    'creditscore': 'credit_score',
    'monthsemployed': 'months_employed',
    'numcreditlines': 'num_credit_lines',
    'interestrate': 'interest_rate',
    'loanterm': 'loan_term',
    'dtiratio': 'dti_ratio',
    'employmenttype': 'employment_type', 
    'maritalstatus': 'marital_status',
    'hasmortgage': 'has_mortgage',
    'hasdependents': 'has_dependents',
    'loanpurpose': 'loan_purpose',
    'hascosigner': 'has_cosigner'
}

BINARY_COLS = ['has_mortgage', 'has_dependents', 'has_cosigner']
BINARY_MAP  = {'Yes': 1, 'No': 0}

OUTPUT_COL_ORDER = [
    'persona_name', 'loan_id', 'age', 'income',
    'loan_amount', 'credit_score', 'months_employed',
    'years_employed', 'num_credit_lines', 'interest_rate',
    'loan_term', 'loan_term_years', 'default',
    'dti_ratio', 'dti_ratio_pct', 'education',
    'employment_type', 'marital_status', 'has_mortgage',
    'has_dependents', 'loan_purpose', 'has_cosigner']


# functions definitions
def extract(path: str) -> pd.DataFrame:
    '''
    Load raw CSV from disk.
    '''
    df = pd.read_csv(path)

    return df

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Set columns to lowercase and rename into snake_case format.
    '''
    df = df.copy()
    df.columns = df.columns.str.lower()
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)
    return df

def validate(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Execute data quality checks
    '''
    #  checks if the required columns are in the dataframe
    required = set(COLUMN_RENAME_MAP.values()) | {'age', 'income', 'education', 'default'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns after rename: {missing}")
    
    # checks for Null values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        print("Null values detected:\n", null_cols.to_string())

    # checks if the binary columns contain only Yes/No values
    for col in BINARY_COLS:
        unexpected = set(df[col].unique()) - set(BINARY_MAP.keys())
        if unexpected:
            raise ValueError(
                f"Column '{col}' contains unexpected values: {unexpected}. "
                f"Expected {set(BINARY_MAP.keys())}."
            )
    print('Validation passed')



def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Map Yes/No columns into 0/1 integers 
    for future Machine Learning purposes.
    '''
    
    df = df.copy()
    df[BINARY_COLS] = df[BINARY_COLS].replace(BINARY_MAP)

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Derive new features from existing ones.
    ''' 
    df= df.copy()
    
    df['dti_ratio_pct'] = df['dti_ratio'] * 100
    df['years_employed'] = (df['months_employed'] / 12).round(1)
    df['loan_term_years'] = (df['loan_term'] / 12).astype('int64')

    return df


def assign_persona(row: pd.Series) -> str:
    """
    Assigns a risk label for each persona to a single loan record row.

    Persona hierarchy:
        gig_worker             -  self-employed individual with higher income volatility.
        financially_stretched  -  borrower with high DTI ratio and income below £40_000.
        stable_low_risk        -  borrower with stable income sources and low financial instability.
        joint_borrower         -  customer with joint lending products (loan/mortgages).
        bonus_income           -  individual with several and different income streams.
        broad_income_mix       -  borrower with mid-to-higher income, moderate DTI ratio,
                                  and no obvious high‑risk flags.
        mainstream             -  all other borrowers not yet classified
    """
    if row['employment_type'] == 'self-employed':
        return 'gig_worker'

    elif row['income'] < 40_000 and row['dti_ratio'] > 0.40:
        return 'financially_stretched'

    elif row['income'] > 90_000 and row['dti_ratio'] < 0.25:
        return 'stable_low_risk'

    elif row['has_cosigner'] == 1:
        return 'joint_borrower'

    elif row['income'] > 70_000 and row['years_employed'] > 5:
        return 'bonus_income'
    
    elif (
        90_000 <= row['income'] <= 110_000 and
        0.25   <= row['dti_ratio'] <= 0.40 and
        600    <= row['credit_score'] <= 700 and
        (
            row['years_employed'] >= 3 or
            row['has_mortgage']   == 1 or
            row['has_dependents'] == 1
        )
    ):
        return 'broad_income_mix'

    else:
        return 'mainstream'

def create_persona_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a pre-loaded loans DataFrame, adds a 'persona' column derived
    from borrower risk characteristics, and returns the enriched DataFrame.
    """
    df = df.copy()
    df["persona_name"] = df.apply(assign_persona, axis=1)

    return df

def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Casts columns to their target PostgreSQL columns dtypes.
    """
    df = df.copy()
    dtype_map = {
        "loan_id":           "string",
        "persona_name":      "string",
        "age":               "int64",
        "income":            "int64",
        "loan_amount":       "int64",
        "credit_score":      "int64",
        "months_employed":   "int64",
        "years_employed":    "float32",
        "num_credit_lines":  "int64",
        "interest_rate":     "float64",
        "loan_term":         "int64",
        "loan_term_years":   "int64",
        "dti_ratio":         "float64",
        "dti_ratio_pct":     "float64",
        "default":           "int8",
        "education":         "string",
        "employment_type":   "string",
        "marital_status":    "string",
        "has_mortgage":      "int8",
        "has_dependents":    "int8",
        "loan_purpose":      "string",
        "has_cosigner":      "int8",
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    return df

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns.
    """
    missing = set(OUTPUT_COL_ORDER) - set(df.columns)
    if missing:
        raise ValueError(f"Cannot reorder — columns missing from DataFrame: {missing}")

    return df[OUTPUT_COL_ORDER]


# pipeline
def run_pipeline(input_path: str) -> pd.DataFrame:
    """
    Execute the full ETL pipeline and return the clean DataFrame.
    """

    df = extract(input_path)
    df = standardise_columns(df)
    validate(df)
    df = encode_binary_columns(df)
    df = engineer_features(df)
    df = create_persona_profiles(df)
    df = enforce_dtypes(df)
    df = reorder_columns(df)

    print("=== ETL pipeline complete — rows ready for ingestion ===", len(df))
    
    return df



if __name__ == "__main__":
    df_clean = run_pipeline(INPUT_PATH)
    print(df_clean.head(2))


    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")


    table_name = "loan_applications"
    df_clean.to_sql(table_name, 
                    engine,
                    schema="raw",
                    if_exists='replace',
                    index=False,
    )

    print(f"Data successfuly loaded into table {table_name} in database {DB_NAME}")

