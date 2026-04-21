import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

import pandas as pd


# main config settings
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

_ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)
main_data_path = os.getenv("PLAID_ACCOUNT_FINAL_PATH")

INPUT_PATH  = os.getenv("INPUT_PATH")
PLAID_ACCOUNTS  = os.path.join(main_data_path, "plaid_accounts_cleaned.csv")
PLAID_TRANSACTIONS  = os.path.join(main_data_path, "plaid_transactions_cleaned.csv")

DB_USER     = os.getenv("ADMIN")
DB_PASSWORD = os.getenv("PASSWORD")
DB_HOST     = os.getenv("HOST", "localhost")
DB_PORT     = os.getenv("PORT", "5432")
DB_NAME     = os.getenv("DB_NAME")





def load_loans_data(loans: str = INPUT_PATH) -> pd.DataFrame:
    """
    Loads the loans dataset as `df` and standardizes column names in snake_case.
    """
    df = clean_column_names(pd.read_csv(loans))

    return df

def load_transactions_data(txs_path: str = PLAID_TRANSACTIONS) -> pd.DataFrame:
    """
    Loads the cleaned Plaid API transactions dataset.
    """
    df = pd.read_csv(txs_path)

    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `df` with standardized snake_case column names.
    """
    DELIMITER_PATTERN = r"[_-]" # finds the characters: `_` or `-` 
    CAMELCASE_PATTERN = r"([a-z])([A-Z])" # finds any lowercase and uppercase
    WHITESPACE_PATTERN = r"\s+" # finds all whitespace occurences

    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns
        .str.strip()
        .str.replace(DELIMITER_PATTERN, " ", regex=True)
        .str.replace(CAMELCASE_PATTERN, r"\1 \2", regex=True)
        .str.replace(WHITESPACE_PATTERN, "_", regex=True)
        .str.lower()
    )

    return cleaned






def generate_features(df_transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Derives per-persona statistical features from a cleaned transactions
    DataFrame. 
    Expects columns: persona_name, booking_date, amount.

    Returns a DataFrame indexed by persona_name with the following features:
        avg_monthly_inflow      -  mean monthly positive cashflow
        income_stability_ratio  -  1 minus the coefficient of variation of monthly inflows
                                   (higher = more stable, range 0-1)
        spend_to_income_ratio   -  total outflows / total inflows (lower = healthier)
        months_net_negative_6m  -  number of months where net cashflow was negative
        cash_withdrawal_ratio   -  share of total spend attributed to cash withdrawals
    """
    df = df_transactions.copy()

    df['booking_date'] = pd.to_datetime(df['booking_date'])
    df['month'] = pd.to_datetime(df['booking_date']).dt.month

    inflow = df[df["amount"] > 0]
    outflow = df[df["amount"] < 0]

    # calculate each persona income consistency for the last 3 months
    cutoff_date = df['booking_date'].max() - pd.DateOffset(months=3)
    df_3m = df[df['booking_date'] >= cutoff_date]
    monthly_totals = df_3m[df_3m['amount'] > 0].groupby(['persona_name', "month"])['amount'].sum()

    std_income_3m = monthly_totals.groupby('persona_name').std()
    mean_income_3m = monthly_totals.groupby('persona_name').mean()
    income_consistency = std_income_3m / mean_income_3m


    # calculate each persona monthly inflow
    n_months = df.groupby("persona_name")["month"].nunique()
    avg_monthly_inflow = inflow.groupby("persona_name")["amount"].sum() / n_months

    monthly_inflow_sum = (
        inflow.groupby(['persona_name', 'month'])['amount'].sum()
    )

    inflow_cv = (
    monthly_inflow_sum.groupby(level=0).std() /
    monthly_inflow_sum.groupby(level=0).mean()
    )

    # calculate each persona income stability
    income_stability_ratio = (1 - inflow_cv).clip(lower=0)


    # calculate each persona spend-to-income ratio
    spend_to_income_ratio = (
        outflow.groupby('persona_name')['amount'].sum().abs() / 
        inflow.groupby('persona_name')['amount'].sum()
    )
    

    # assessing each persona negative net cash flow and counting the number of months
    monthly_net = df.groupby(['persona_name', 'month'])['amount'].sum()

    months_net_negative = (
        monthly_net[monthly_net < 0]
        .groupby(level=0)
        .count()
        .rename("months_net_negative_6m")
    )

    # Count the number of months with negative net cash flow for each persona
    cash_keywords = r"ATM|CASH|WITHDRAWAL|WITHDRAW"
    cash_mask = df["remitter"].str.upper().str.contains(cash_keywords, na=False)

    cash_outflow = df[cash_mask & (df["amount"] < 0)]
    cash_withdrawal_ratio = (
        cash_outflow.groupby("persona_name")["amount"].sum().abs() /
        outflow.groupby("persona_name")["amount"].sum().abs()
    ).fillna(0)
    

    
    features = pd.DataFrame({
        "persona_name":           df["persona_name"].unique(),
        "avg_monthly_inflow":     avg_monthly_inflow,
        "income_consistency":     income_consistency,
        "income_stability_ratio": income_stability_ratio,
        "spend_to_income_ratio":  spend_to_income_ratio,
        "months_net_negative_6m": months_net_negative,
        "cash_withdrawal_ratio":  cash_withdrawal_ratio,
    })

    return features


def pipeline() -> pd.DataFrame:
    """
    Orchestrates the full feature derivation workflow:
        1. Load raw loans data and assign personas
        2. Load cleaned transactions data
        3. Merge personas onto transactions
        4. Derive statistical features per persona
    """
    df_transactions = load_transactions_data()
    df_features = generate_features(df_transactions)
    df_features = df_features.round(2)
    df_features = df_features.fillna(0.0)    

    return df_features

if __name__ == "__main__":
    df = pipeline()
    
    df.to_csv('persona_features.csv', index=False)


    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")


    table_name = "persona_features"
    df.to_sql(table_name, 
                    engine,
                    schema="analytics",
                    if_exists='replace',
                    index=False,
    )

    print(f"Data successfuly loaded into table: {table_name} in database: {DB_NAME}")
