# from plaid.model import transaction
import os
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

import pandas as pd

# main config settings
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

_ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

accounts_path = os.getenv("PLAID_ACCOUNT_PATH")
transactions_path = os.getenv("PLAID_TRANSACTIONS_PATH")

accounts_final_path = os.getenv("PLAID_ACCOUNT_FINAL_PATH")
transactions_final_path = os.getenv("PLAID_TRANSACTIONS_FINAL_PATH")

DB_USER     = os.getenv("ADMIN")
DB_PASSWORD = os.getenv("PASSWORD")
DB_HOST     = os.getenv("HOST", "localhost")
DB_PORT     = os.getenv("PORT", "5432")
DB_NAME     = os.getenv("DB_NAME")



def load_tx_data() -> pd.DataFrame:
    df_tx = pd.read_csv(transactions_path)

    return df_tx

def load_acc_data() -> pd.DataFrame:
    df_acc = pd.read_csv(accounts_path)

    return df_acc

def check_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.dropna(axis=1)

    return df_clean


def clean_df_tx(df:pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            'accountid': 'account_id',
            'currency': 'currency',
            'bookingdate': 'booking_date',
            'valuedate': 'transaction_date',
            'remittanceinfo': 'remitter',
            'direction': 'transaction_type',
            }
    )


    df["booking_date"] = pd.to_datetime(df["booking_date"], format="%Y-%m-%d").dt.date
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y-%m-%d").dt.date

    df['currency'] = "GBP"
    df['transaction_type'] = df['transaction_type'].str.capitalize()

    df = df[[
        'persona_name', 'booking_date',
        'account_id', 'transaction_date',
        'amount', 'currency', 'remitter'
        ]]

    return df

def clean_df_acc(accounts:pd.DataFrame) -> pd.DataFrame:
    df = accounts

    df.drop(["accounttype", "status"], axis=1, inplace=True)
    df['currency'] = "GBP"

    df = df.rename(
        columns={
            'currency': 'currency',
            'accountid': 'account_id',
            "name": "account_type",
            }
    )

    return df


if __name__ == "__main__":
    df_tx = load_tx_data()
    df_acc = load_acc_data()

    df_txs_no_nulls = check_nulls(df_tx)
    df_acc_no_nulls = check_nulls(df_acc)

    df_txs_clean = clean_df_tx(df_txs_no_nulls)
    df_acc_clean = clean_df_acc(df_acc_no_nulls)

    output_path_txs = os.path.join(transactions_final_path, "plaid_transactions_cleaned.csv")
    output_path_acc = os.path.join(accounts_final_path, "plaid_accounts_cleaned.csv")

    df_txs_clean.to_csv(output_path_txs, index=False)
    df_acc_clean.to_csv(output_path_acc, index=False)


   

    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")


    table_name = "transactions"
    table_name_2 = "bank_accounts"


    df_txs_clean.to_sql(table_name, 
                    engine,
                    schema="raw",
                    if_exists='replace',
                    index=False,
    )

    df_acc_clean.to_sql(table_name_2, 
                    engine,
                    schema="raw",
                    if_exists='replace',
                    index=False,
    )


    print(f"Data successfuly loaded into table: {table_name} in database: {DB_NAME}")
    print(f"Data successfuly loaded into table: {table_name_2} in database: {DB_NAME}")
