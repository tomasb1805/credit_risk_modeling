import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

# main config settings
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

TABLE = os.getenv("MODEL_TABLE_PATH")

def load_table(table_path: str = TABLE) -> pd.DataFrame:
    """
    Load the database table to be preprocessed.
    """
    df = pd.read_csv(table_path)

    return df

def
