# create_schema.py
import sqlite3
import os

def create_schema(db_path):
    """Create the database schema for LLM experiments with hierarchical traits"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create Experiments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id INTEGER PRIMARY KEY,
        date_time DATETIME,
        model_version TEXT,
        quantization_level TEXT,
        description TEXT
    )
    ''')
    
    # Create Traits table with parent_trait_id for hierarchical structure
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS traits (
        trait_id INTEGER PRIMARY KEY,
        domain TEXT,                      -- "Personality", "Political", "Moral"
        specificity TEXT,                 -- "Broad", "Specific"
        trait_name TEXT,
        trait_description TEXT,
        parent_trait_id INTEGER,          -- References another trait_id for specific traits
        FOREIGN KEY (parent_trait_id) REFERENCES traits(trait_id)
    )
    ''')
    
    # Create Steering Vectors table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS steering_vectors (
        vector_id INTEGER PRIMARY KEY,
        trait_id INTEGER,
        vector_data BLOB,
        vector_dimensions INTEGER,
        layer_applied INTEGER,
        FOREIGN KEY (trait_id) REFERENCES traits(trait_id)
    )
    ''')
    
    # Create Prompts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prompts (
        prompt_id INTEGER PRIMARY KEY,
        prompt_text TEXT,
        prompt_category TEXT
    )
    ''')
    
    # Create Outputs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS outputs (
        output_id INTEGER PRIMARY KEY,
        experiment_id INTEGER,
        prompt_id INTEGER,
        trait_id INTEGER,
        steering_strength REAL,
        output_text TEXT,
        generation_time REAL,
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
        FOREIGN KEY (prompt_id) REFERENCES prompts(prompt_id),
        FOREIGN KEY (trait_id) REFERENCES traits(trait_id)
    )
    ''')
    
    # Create Evaluations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluations (
        evaluation_id INTEGER PRIMARY KEY,
        output_id INTEGER,
        evaluator TEXT,
        trait_id INTEGER,
        trait_score REAL,
        confidence REAL,
        additional_metrics TEXT,
        FOREIGN KEY (output_id) REFERENCES outputs(output_id),
        FOREIGN KEY (trait_id) REFERENCES traits(trait_id)
    )
    ''')
    
    conn.commit()
    print(f"Schema created successfully in {db_path}")
    return conn

if __name__ == "__main__":
    create_schema('results/database/experiments.db')