from sqlalchemy import create_engine
import pandas as pd

# PostgreSQL connection details
DB_URL = "postgresql://postgres:5492460@localhost:5432/10-academy"

def get_database_connection():
    """
    Creates and returns a SQLAlchemy engine for connecting to the PostgreSQL database.
    """
    return create_engine(DB_URL)

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
