# import_traits.py
import pandas as pd
import sqlite3
from pathlib import Path

def import_traits_from_csv(csv_path, db_path='results/database/experiments.db'):
    """Import traits from CSV file with hierarchical relationships"""
    # Read CSV
    traits_df = pd.read_csv(csv_path)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # First pass: Add broad traits
    broad_traits = traits_df[traits_df['specificity'] == 'Broad']
    trait_ids = {}  # To store trait_id by name
    
    for _, row in broad_traits.iterrows():
        cursor.execute("""
            INSERT INTO traits (domain, specificity, trait_name, trait_description, parent_trait_id)
            VALUES (?, ?, ?, ?, NULL)
        """, (row['domain'], row['specificity'], row['trait_name'], row['description']))
        trait_id = cursor.lastrowid
        trait_ids[row['trait_name']] = trait_id
    
    # Second pass: Add specific traits with parent references
    specific_traits = traits_df[traits_df['specificity'] == 'Specific']
    
    for _, row in specific_traits.iterrows():
        parent_id = trait_ids.get(row['parent_trait'], None)
        cursor.execute("""
            INSERT INTO traits (domain, specificity, trait_name, trait_description, parent_trait_id)
            VALUES (?, ?, ?, ?, ?)
        """, (row['domain'], row['specificity'], row['trait_name'], row['description'], parent_id))
    
    conn.commit()
    conn.close()
    print(f"Imported {len(traits_df)} traits from {csv_path}")

if __name__ == "__main__":
    import_traits_from_csv('scripts/database_management/trait_structure_management/trait_master_db.csv')