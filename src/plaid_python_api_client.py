# Python libraries
import os
import time
from pathlib import Path
from datetime import date, timedelta
from dotenv import load_dotenv

import pandas as pd

# Plaid API libraries
from plaid.api.plaid_api import PlaidApi
from plaid.exceptions import ApiException
from plaid.model.products import Products
from plaid import ApiClient, Configuration, Environment
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.sandbox_public_token_create_request import SandboxPublicTokenCreateRequest

# dictionary for Plaid Test Personas and information to pull from these accounts:

PERSONAS = [
    {"persona_name": "stable_low_risk",
     "plaid_username": "user_credit_profile_excellent",
      "password": "test"},

    {"persona_name": "financially_stretched",
     "plaid_username": "user_credit_profile_poor",
      "password": "test"},

    {"persona_name": "gig_worker",
     "plaid_username": "user_credit_profile_good",
      "password": "test"},

    {"persona_name": "bonus_income",
     "plaid_username": "user_credit_bonus",
      "password": "test"},
    
    # this specific Plaid ready-made persona specifically requires "{}" as a password,
    # the other personas can have any password as a string
    {"persona_name": "broad_income_mix", 
    "plaid_username": "user_bank_income",
     "password": "{}"}, 

    {"persona_name": "joint_borrower",
     "plaid_username": "user_credit_joint_account",
      "password": "test"}
]

# this line establishes a script-oriented flow (the "try" branch) or notebook-style environment (the "except" branch)
# this approach makes sure the script will work for different workflow choices
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

# established the .env path and loads it to the ENV_PATH variable to be fed into load_dotenv
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SANDBOX_SECRET")
PLAID_API_RAW_PATH = os.getenv("PLAID_API_RAW_PATH")

# basic Plaid Sandbox API configuration
configuration = Configuration(
    host=Environment.Sandbox,
    api_key={
        "clientId": PLAID_CLIENT_ID,
        "secret": PLAID_SECRET,
        "plaidVersion": "2020-09-14",
    },
)
client = PlaidApi(ApiClient(configuration))


# API requests format for each persona

def poll_with_retries(request_fn, max_retries=10, delay_seconds=1.0):
    """Call a Plaid request fn, retrying on PRODUCT_NOT_READY."""
    for attempt in range(1, max_retries + 1):
        try:
            return request_fn()
        except ApiException as e:
            body = getattr(e, "body", "") or ""
            if "PRODUCT_NOT_READY" in str(body):
                print(f"PRODUCT_NOT_READY (attempt {attempt}), sleeping {delay_seconds}s...")
                time.sleep(delay_seconds)
                continue
            raise
    raise RuntimeError("Ran out of retries while waiting for product to be ready")

def normalise_plaid_accounts(accounts, persona_name):
    rows = []
    for a in accounts:
        rows.append({
            "persona_name": persona_name,
            "accountid": a["account_id"],
            "currency": a["balances"].get("iso_currency_code", "USD"),
            "accounttype": a.get("subtype"),
            "name": a.get("name"),
            "owner": None,
            "status": "enabled",
        })
    return pd.DataFrame(rows)

def normalise_plaid_transactions(transactions, persona_name):
    rows = []
    for t in transactions:
        amount = float(t["amount"])
        # Default Plaid convetions treat debits amounts as a positive number and credits as a negative number
        # Hence the negative in front of "amount": to swap the sign and make the number more user-friendly/more readable
        signed_amount = -amount
        rows.append({
            "persona_name": persona_name,
            "transactionid": t["transaction_id"],
            "accountid": t["account_id"],
            "bookingdate": t["date"],
            "valuedate": t.get("authorized_date") or t["date"],
            "amount": signed_amount,
            "currency": t.get("iso_currency_code", "USD"),
            "creditorname": t.get("merchant_name") or "",
            "debtorname": "",
            "remittanceinfo": t.get("name") or "",
            "direction": "debit",   # with the sign convention described above
        })
    return pd.DataFrame(rows)


# main API call loop

all_accounts = []
all_transactions = []

for p in PERSONAS:
    persona_name = p["persona_name"]
    username = p["plaid_username"]
    password = p["password"]
    print(f"\n=== Persona: {persona_name} ({username}) ===")

    # Create sandbox public_token
    sandbox_req = SandboxPublicTokenCreateRequest(
        institution_id="ins_109508",
        initial_products=[Products("transactions")],
        options={
            "override_username": username,
            "override_password": password,
        },
    )

    sandbox_res = client.sandbox_public_token_create(sandbox_req)
    public_token = sandbox_res["public_token"]

    # Exchange public_token for access_token
    exchange_req = ItemPublicTokenExchangeRequest(public_token=public_token)
    exchange_res = client.item_public_token_exchange(exchange_req)
    access_token = exchange_res["access_token"]

    # Fetch persona accounts
    acc_req = AccountsGetRequest(access_token=access_token)
    acc_res = client.accounts_get(acc_req)
    accounts = acc_res["accounts"]
    print(f"  Accounts: {len(accounts)}")

    df_acc = normalise_plaid_accounts(accounts, persona_name)
    all_accounts.append(df_acc)

    # Fetch transactions with retry & sleep timer
    start_date = date.today() - timedelta(days=120)
    end_date = date.today()

    tx_req = TransactionsGetRequest(
        access_token=access_token,
        start_date=start_date,
        end_date=end_date,
        options=TransactionsGetRequestOptions()
    )

    def do_tx_request():
        return client.transactions_get(tx_req)

    tx_res = poll_with_retries(do_tx_request, max_retries=10, delay_seconds=1.0)
    transactions = tx_res["transactions"]
    print(f"  Transactions: {len(transactions)}")

    df_tx = normalise_plaid_transactions(transactions, persona_name)
    all_transactions.append(df_tx)

#  combine into a single DataFrame:
df_accounts = pd.concat(all_accounts, ignore_index=True) if all_accounts else pd.DataFrame()
df_transactions = pd.concat(all_transactions, ignore_index=True) if all_transactions else pd.DataFrame()

print("\nFinal shapes:")
print("df_accounts:", df_accounts.shape)
print("df_transactions:", df_transactions.shape)

accounts_api_raw_path = os.path.join(PLAID_API_RAW_PATH, "persona_accounts_api_stream.csv")
txs_api_raw_path = os.path.join(PLAID_API_RAW_PATH, "persona_txs_api_stream.csv")

df_accounts.to_csv(accounts_api_raw_path, sep=',', index=False)
df_transactions.to_csv(txs_api_raw_path, sep=',', index=False)

