from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_database_connection():
    """
    Creates and returns a SQLAlchemy engine for connecting to the PostgreSQL database.
    Retrieves database credentials from the .env file.
    """
    db_url = os.getenv("DB_URL_DASHBOARD")
    if not db_url:
        raise EnvironmentError("DB_URL_DASHBOARD environment variable not set.")
    
    return create_engine(db_url)

def fetch_data(query):
    """
    Executes a SQL query and returns the result as a pandas DataFrame.
    :param query: The SQL query to execute.
    """
    engine = get_database_connection()
    try:
        return pd.read_sql_query(query, engine)
    finally:
        engine.dispose()
