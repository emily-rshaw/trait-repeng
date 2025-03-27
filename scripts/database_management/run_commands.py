import sqlite3

db_path = "results/database/experiments.db"

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    
    # If 'target_response' is stored as text '1'/'5':
    cursor.execute("""
        UPDATE prompts
        SET target_response = 'I would rate myself a 1'
        WHERE target_response = '1'
    """)
    
    cursor.execute("""
        UPDATE prompts
        SET target_response = 'I would rate myself a 5'
        WHERE target_response = '5'
    """)
    
    conn.commit()
