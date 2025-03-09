import sqlite3
import os

# Initialize the database
db_path = 'results/database/experiments.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print(f"Database created at {db_path}")