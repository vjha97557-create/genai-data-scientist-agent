import re
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from llm_agent import ask_llm

DB_FILE = "sales.db"
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|truncate|grant|vacuum)\b", re.I)


def save_to_db(df, table="sales"):
    eng = create_engine(f"sqlite:///{DB_FILE}")
    df.to_sql(table, eng, if_exists="replace", index=False)
    eng.dispose()


def _read_only_engine():
    # SQLite read-only mode. With MySQL, create a read-only database user and change the URL.
    return create_engine(f"sqlite:///file:{DB_FILE}?mode=ro&uri=true")


def get_schema():
    eng = _read_only_engine()
    insp = inspect(eng)
    out = "\n".join(
        f"Table {t}: " + ", ".join(f'"{c["name"]}" ({c["type"]})' for c in insp.get_columns(t))
        for t in insp.get_table_names())
    eng.dispose()
    return out


def validate_sql(sql):
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("Only a single query is allowed.")
    if not re.match(r"(?i)^(select|with)\b", sql):
        raise ValueError("Only SELECT statements are allowed.")
    if FORBIDDEN.search(sql):
        raise ValueError("The query contains a blocked keyword.")
    if not re.search(r"(?i)\blimit\b", sql):
        sql += " LIMIT 500"
    return sql


def question_to_sql(question):
    prompt = f"""Write one SQLite SELECT query for this schema.
{get_schema()}
Rules: wrap column names in double quotes (e.g. "Sub-Category"). Dates are stored as text 'YYYY-MM-DD HH:MM:SS'.
Question (may be in English, Hindi or Hinglish): {question}
Return ONLY the SQL, no explanation, no markdown."""
    sql = ask_llm(prompt, enforce_english=False).replace("```sql", "").replace("```", "").strip()
    return validate_sql(sql)


def run_sql(sql):
    eng = _read_only_engine()
    try:
        with eng.connect() as conn:
            return pd.read_sql(text(validate_sql(sql)), conn)
    finally:
        eng.dispose()
