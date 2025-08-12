# backend/community.py
"""
Simple local community/crowdsourcing backend.
Stores submissions to backend/data/community_submissions.csv

Functions:
  - add_submission(name, email, role, submission_type, content, attached_filename=None)
  - list_submissions() -> pandas.DataFrame
"""

from pathlib import Path
import pandas as pd
from datetime import datetime
import csv
import os

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "community_submissions.csv"

# Ensure CSV exists with header
if not DB_PATH.exists():
    df_init = pd.DataFrame(columns=[
        "timestamp", "name", "email", "role", "submission_type", "content", "attached_filename"
    ])
    df_init.to_csv(DB_PATH, index=False)

def add_submission(name: str, email: str, role: str, submission_type: str, content: str, attached_filename: str = None):
    """
    Append a submission row to the CSV DB.
    """
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "name": name,
        "email": email,
        "role": role,
        "submission_type": submission_type,
        "content": content,
        "attached_filename": attached_filename or ""
    }
    # Append as a new CSV row safely
    with open(DB_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)
    return True

def list_submissions(limit: int = 200) -> pd.DataFrame:
    """
    Read CSV and return DataFrame. Limit rows to `limit` most recent.
    """
    try:
        df = pd.read_csv(DB_PATH)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp", ascending=False)
        if limit:
            df = df.head(limit)
        return df.reset_index(drop=True)
    except Exception:
        # if something went wrong, return empty df with schema
        return pd.DataFrame(columns=[
            "timestamp", "name", "email", "role", "submission_type", "content", "attached_filename"
        ])
