import sqlite3

db_path = "results/database/experiments.db"

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.execute("ALTER TABLE runs ADD COLUMN shuffle_index INTEGER;")
    cursor.execute("ALTER TABLE runs ADD COLUMN val_size INTEGER;")
    cursor.execute("ALTER TABLE runs ADD COLUMN test_size INTEGER;")
    conn.commit()
