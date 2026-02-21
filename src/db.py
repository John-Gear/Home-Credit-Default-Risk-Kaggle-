import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH_TRAIN = BASE_DIR / "data" / "application_train.db"
DB_PATH_TEST  = BASE_DIR / "data" / "for_predict" / "application_test.db"

def read_sql_train(query: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH_TRAIN))
    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def read_sql_test(query: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH_TEST))
    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()