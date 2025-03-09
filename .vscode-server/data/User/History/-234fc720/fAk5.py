# db_helpers.py
import sqlite3
import json
import numpy as np
from datetime import datetime

def get_connection(db_path='results/database/experiments.db'):
    """Connect to the database"""
    return sqlite3.connect(db_path)

def add_experiment(conn, model_version, quantization_level, description):
    """Add a new experiment run to the database"""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO experiments (date_time, model_version, quantization_level, description)
        VALUES (?, ?, ?, ?)
    """, (datetime.now(), model_version, quantization_level, description))
    conn.commit()
    return cursor.lastrowid

def add_trait(conn, domain, specificity, trait_name, trait_description, parent_trait_id=None):
    """Add a trait to the database
    
    Args:
        conn: Database connection
        domain: "Personality", "Political", or "Moral"
        specificity: "Broad" or "Specific"
        trait_name: Name of the trait
        trait_description: Description of the trait
        parent_trait_id: ID of the parent trait (if this is a specific trait)
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO traits (domain, specificity, trait_name, trait_description, parent_trait_id)
        VALUES (?, ?, ?, ?, ?)
    """, (domain, specificity, trait_name, trait_description, parent_trait_id))
    conn.commit()
    return cursor.lastrowid

def get_trait_id(conn, trait_name):
    """Get the trait_id for a given trait name"""
    cursor = conn.cursor()
    cursor.execute("SELECT trait_id FROM traits WHERE trait_name = ?", (trait_name,))
    result = cursor.fetchone()
    return result[0] if result else None

def get_trait_details(conn, trait_id):
    """Get details for a trait including its parent if applicable"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.trait_id, t.domain, t.specificity, t.trait_name, 
               t.trait_description, t.parent_trait_id, 
               p.trait_name as parent_name
        FROM traits t
        LEFT JOIN traits p ON t.parent_trait_id = p.trait_id
        WHERE t.trait_id = ?
    """, (trait_id,))
    return cursor.fetchone()

def get_specific_traits_for_parent(conn, parent_trait_id):
    """Get all specific traits belonging to a parent trait"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT trait_id, trait_name, trait_description
        FROM traits
        WHERE parent_trait_id = ?
    """, (parent_trait_id,))
    return cursor.fetchall()

def add_prompt(conn, prompt_text, prompt_category):
    """Add a prompt to the database"""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prompts (prompt_text, prompt_category)
        VALUES (?, ?)
    """, (prompt_text, prompt_category))
    conn.commit()
    return cursor.lastrowid

def add_steering_vector(conn, trait_id, vector_data, layer_applied):
    """Add a steering vector to the database"""
    cursor = conn.cursor()
    # Convert numpy array to binary
    vector_blob = vector_data.tobytes()
    vector_dimensions = len(vector_data)
    
    cursor.execute("""
        INSERT INTO steering_vectors (trait_id, vector_data, vector_dimensions, layer_applied)
        VALUES (?, ?, ?, ?)
    """, (trait_id, vector_blob, vector_dimensions, layer_applied))
    conn.commit()
    return cursor.lastrowid

def add_output(conn, experiment_id, prompt_id, trait_id, steering_strength, output_text, generation_time):
    """Add a model output to the database"""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO outputs (experiment_id, prompt_id, trait_id, steering_strength, output_text, generation_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (experiment_id, prompt_id, trait_id, steering_strength, output_text, generation_time))
    conn.commit()
    return cursor.lastrowid

def add_evaluation(conn, output_id, evaluator, trait_id, trait_score, confidence, additional_metrics=None):
    """Add an evaluation of an output to the database"""
    cursor = conn.cursor()
    if additional_metrics:
        additional_metrics = json.dumps(additional_metrics)
    
    cursor.execute("""
        INSERT INTO evaluations (output_id, evaluator, trait_id, trait_score, confidence, additional_metrics)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (output_id, evaluator, trait_id, trait_score, confidence, additional_metrics))
    conn.commit()
    return cursor.lastrowid

def get_vector(conn, vector_id):
    """Retrieve a steering vector from the database"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT vector_data, vector_dimensions FROM steering_vectors 
        WHERE vector_id = ?
    """, (vector_id,))
    result = cursor.fetchone()
    
    if result:
        vector_blob, dimensions = result
        # Convert binary back to numpy array
        return np.frombuffer(vector_blob, dtype=np.float32)
    return None