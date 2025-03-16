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
        model_name TEXT,
        quantization_level TEXT,
        trait_id INTEGER,
        trait_max_or_min TEXT,
        description TEXT,
        FOREIGN KEY (trait_id) REFERENCES traits(trait_id)
    )
    ''')
    
    # table for individual runs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS runs (
        run_id INTEGER PRIMARY KEY,
        experiment_id INTEGER,
        target_vector_id INTEGER,
        duration_in_seconds INTEGER,
        seed INTEGER,
        max_new_tokens INTEGER,
        num_samples INTEGER,
        max_seq_len INTEGER,
        source_layer_idx INTEGER,
        target_layer_idx INTEGER,
        num_factors INTEGER,
        forward_batch_size INTEGER,
        backward_batch_size INTEGER,
        factor_batch_size INTEGER,
        num_eval INTEGER,
        system_prompt TEXT,
        dim_output_projection INTEGER,
        beta REAL,
        max_iters INTEGER,
        target_ratio REAL,
        input_scale REAL,
        run_description TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
        FOREIGN KEY (target_vector_id) REFERENCES target_vectors(target_vector_id)
    )
    ''')

    # table for sets of prompts (e.g.) max_c5_self-discipline.csv
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompt_sets (
            prompt_set_id INTEGER PRIMARY KEY,
            trait_id INTEGER,
            set_name TEXT,
            set_closed_or_open_ended TEXT,
            set_description TEXT,
            FOREIGN KEY (trait_id) REFERENCES traits(trait_id)
        )
    ''')

    # Join table for many-to-many between experiments and prompt sets
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiment_prompt_sets (
            experiment_id INTEGER,
            prompt_set_id INTEGER,
            PRIMARY KEY (experiment_id, prompt_set_id),
            FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
            FOREIGN KEY (prompt_set_id) REFERENCES prompt_sets(prompt_set_id)
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

    # each prompt belongs to prompt set
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompts (
            prompt_id INTEGER PRIMARY KEY,
            prompt_set_id INTEGER,
            prompt_text TEXT,
            target_response TEXT,
            FOREIGN KEY (prompt_set_id) REFERENCES prompt_sets(prompt_set_id)
        )
    ''')
    
    # Create Steering Vectors table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS steering_vectors (
            vector_id INTEGER PRIMARY KEY,
            created_by_run_id INTEGER,                  -- the run that created this vector
            vector_rank_score REAL,
            vector_rank_index INTEGER,            -- technically probably unnormalised...alas 
            vector_data BLOB,                -- storing the actual vector
            is_random INTEGER DEFAULT 0,     -- 0/1, if vector is random/used for control
            FOREIGN KEY (created_by_run_id) REFERENCES runs(run_id)
        )
    ''')

    # Create target vecs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS target_vectors (
            target_vector_id INTEGER PRIMARY KEY,
            token_a TEXT,                 -- e.g., "Sorry"
            token_b TEXT,                 -- e.g., "Sure"
            token_a_id INTEGER,           -- e.g., the tokenizer ID for "Sorry"
            token_b_id INTEGER,           -- e.g., the tokenizer ID for "Sure"
            vector_data BLOB,             -- store the actual difference vector
            creation_method TEXT,         -- "difference of final embeddings", etc.
            notes TEXT                    -- any optional info
       )
    ''')
    
    # Create Outputs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS outputs (
            output_id INTEGER PRIMARY KEY,
            run_id INTEGER,                  -- which run was used
            prompt_id INTEGER,               -- which prompt
            vector_id INTEGER,               -- which steering vector
            output_text TEXT,                -- generated text
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (prompt_id) REFERENCES prompts(prompt_id),
            FOREIGN KEY (vector_id) REFERENCES steering_vectors(vector_id)
        )
    ''')
    
    # eval table will also be needed, but can be created later
    
    conn.commit()
    print(f"Schema created successfully in {db_path}")
    return conn

if __name__ == "__main__":
    create_schema('results/database/experiments.db')