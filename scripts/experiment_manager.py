#!/usr/bin/env python3
"""
Experiment Manager

- Create an experiment in the 'experiments' table.
- Create runs in the 'runs' table, passing all run parameters explicitly.
- Call run_executor.execute_run(run_id) to actually perform the run and update duration_in_seconds.
"""

import os
import sqlite3
import datetime
import torch

# 1) Import the run_executor
from run_executor import execute_run as run_dct
from run_executor import prepare_model_and_tokenizer as prep_mod_and_tok

DB_PATH = "results/database/experiments.db"


class ExperimentManager:
    def __init__(self, db_path=DB_PATH):
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    #########################################################################
    #                   PROMPT METHODS
    #########################################################################
    
    # 1) create_prompt_set
    def create_prompt_set(self, trait_id, set_name, set_closed_or_open_ended, set_description):
        sql = """
        INSERT INTO prompt_sets (
            trait_id,
            set_name,
            set_closed_or_open_ended,
            set_description
        ) VALUES (?, ?, ?, ?)
        """
        self.cursor.execute(sql, (trait_id, set_name, set_closed_or_open_ended, set_description))
        self.conn.commit()
        return self.cursor.lastrowid

    # 2) add_prompt_to_set (with optional target_response)
    def add_prompt_to_set(self, prompt_set_id, prompt_text, target_response=None):
        sql = """
        INSERT INTO prompts (
            prompt_set_id,
            prompt_text,
            target_response
        ) VALUES (?, ?, ?)
        """
        self.cursor.execute(sql, (prompt_set_id, prompt_text, target_response))
        self.conn.commit()
        return self.cursor.lastrowid

    # 3) link_experiment_prompt_set
    def link_experiment_prompt_set(self, experiment_id, prompt_set_id):
        sql = """
        INSERT OR IGNORE INTO experiment_prompt_sets (
            experiment_id,
            prompt_set_id
        ) VALUES (?, ?)
        """
        self.cursor.execute(sql, (experiment_id, prompt_set_id))
        self.conn.commit()

    # 4) get_prompt_sets_for_experiment
    def get_prompt_sets_for_experiment(self, experiment_id):
        sql = """
        SELECT ps.*
        FROM prompt_sets ps
        JOIN experiment_prompt_sets eps 
          ON ps.prompt_set_id = eps.prompt_set_id
        WHERE eps.experiment_id = ?
        """
        rows = self.cursor.execute(sql, (experiment_id,)).fetchall()
        return rows  # or parse into a list of dicts

    # 5) get_prompts_for_prompt_set
    def get_prompts_for_prompt_set(self, prompt_set_id):
        sql = """
        SELECT prompt_text, target_response
        FROM prompts
        WHERE prompt_set_id = ?
        """
        rows = self.cursor.execute(sql, (prompt_set_id,)).fetchall()
        # Each row has 'prompt_text' and 'target_response'
        return [(r["prompt_text"], r["target_response"]) for r in rows]
    
    def create_target_vector(self, token_a, token_b, tokenizer, model, creation_method="diff", notes=""):
        """
        Creates a target vector entry in the 'target_vectors' table.

        By default, 'creation_method="diff"' will compute 
        model.lm_head.weight.data[token_b] - model.lm_head.weight.data[token_a]
        and store it as vector_data (a BLOB).

        :returns: the newly created target_vector_id
        """
        # 1) Convert tokens to IDs
        token_a_ids = tokenizer.encode(token_a, add_special_tokens=False)
        token_b_ids = tokenizer.encode(token_b, add_special_tokens=False)
        if not token_a_ids or not token_b_ids:
            raise ValueError(f"Could not encode tokens '{token_a}' or '{token_b}'")

        token_a_id = token_a_ids[0]
        token_b_id = token_b_ids[0]

        # 2) Optionally compute the difference vector
        target_vec = None
        if creation_method == "diff":
            with torch.no_grad():
                # Grab all embeddings in lm_head for phrase A
                emb_a = model.lm_head.weight.data[token_a_ids, :]  
                # shape: (#tokens_in_A, d_model)
                # Average them to get a single vector for the entire phrase
                emb_a_mean = emb_a.mean(dim=0)  
                # shape: (d_model,)

                # Same for phrase B
                emb_b = model.lm_head.weight.data[token_b_ids, :]
                emb_b_mean = emb_b.mean(dim=0)

                # Finally, the difference vector
                target_vec = emb_b_mean - emb_a_mean  # shape: (d_model,)
        else:
            # If you have other ways to create the vector, handle them here
            target_vec = torch.zeros(model.config.hidden_size)

        # 3) Convert to BLOB
        vector_blob = target_vec.detach().cpu().numpy().tobytes()

        # 4) Insert row
        sql = """
        INSERT INTO target_vectors (
            token_a,
            token_b,
            token_a_id,
            token_b_id,
            vector_data,
            creation_method,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(sql, (
            token_a,
            token_b,
            token_a_id,
            token_b_id,
            vector_blob,
            creation_method,
            notes
        ))
        self.conn.commit()
        return self.cursor.lastrowid


    #########################################################################
    #                   EXPERIMENT METHODS
    #########################################################################

    def create_experiment(
        self,
        model_name,
        quantization_level,
        trait_id,
        trait_max_or_min,
        description
    ):
        """
        Insert a new experiment into the 'experiments' table and return its ID.
        """
        date_time = datetime.datetime.now().isoformat()
        sql = """
        INSERT INTO experiments (
            date_time,
            model_name,
            quantization_level,
            trait_id,
            trait_max_or_min,
            description
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(
            sql,
            (
                date_time,
                model_name,
                quantization_level,
                trait_id,
                trait_max_or_min,
                description
            )
        )
        self.conn.commit()
        return self.cursor.lastrowid

    #########################################################################
    #                   RUN METHODS
    #########################################################################

    def create_run(self, experiment_id, run_params):
        sql = """
        INSERT INTO runs (
            experiment_id,
            target_vector_id,
            seed,
            max_new_tokens,
            num_samples,
            max_seq_len,
            source_layer_idx,
            target_layer_idx,
            num_factors,
            forward_batch_size,
            backward_batch_size,
            factor_batch_size,
            num_eval,
            system_prompt,
            dim_output_projection,
            beta,
            max_iters,
            target_ratio,
            input_scale,
            shuffle_index,                -- new
            val_size,                     -- new
            test_size,                    -- new
            run_description
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = (
            experiment_id,
            run_params.get("target_vector_id", None),
            run_params["seed"],
            run_params["max_new_tokens"],
            run_params["num_samples"],
            run_params["max_seq_len"],
            run_params["source_layer_idx"],
            run_params["target_layer_idx"],
            run_params["num_factors"],
            run_params["forward_batch_size"],
            run_params["backward_batch_size"],
            run_params["factor_batch_size"],
            run_params["num_eval"],
            run_params["system_prompt"],
            run_params["dim_output_projection"],
            run_params["beta"],
            run_params["max_iters"],
            run_params["target_ratio"],
            run_params["input_scale"],

            # Provide default or actual values from run_params
            run_params["shuffle_index"],
            run_params["val_size"],
            run_params["test_size"],

            run_params.get("run_description", "")
        )
        self.cursor.execute(sql, data)
        self.conn.commit()
        return self.cursor.lastrowid


    def create_multiple_runs(self, experiment_id, list_of_run_params):
        """
        Accepts a list of dictionaries (each dictionary is run_params for one run).
        Inserts each run and returns a list of run_ids.
        """
        run_ids = []
        for rp in list_of_run_params:
            run_id = self.create_run(experiment_id, rp)
            run_ids.append(run_id)
        return run_ids

    def execute_run(self, run_id):
        """
        Instead of a placeholder, we call the run_executor's function.
        """
        print(f"[ExperimentManager] Now executing run {run_id} using run_executor...")
        result = run_dct(run_id, db_path=DB_PATH)
        print(f"[ExperimentManager] Execution result: {result}")

    def delete_experiment(self, experiment_id):
        """
        Helper method to remove an experiment and its runs,
        if you don't have ON DELETE CASCADE in your schema.
        """
        self.cursor.execute("DELETE FROM runs WHERE experiment_id = ?", (experiment_id,))
        self.cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
        self.conn.commit()

    def delete_run(self, run_id):
        """
        Helper method to remove a single run.
        """
        self.cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()

def import_prompt_set_from_csv(mgr, csv_path, 
                               trait_id=None, 
                               set_name="ImportedSet", 
                               set_closed_or_open_ended="open", 
                               set_description="Imported from CSV"):
    import csv

    # 1) Create the prompt set
    p_set_id = mgr.create_prompt_set(
        trait_id=trait_id,
        set_name=set_name,
        set_closed_or_open_ended=set_closed_or_open_ended,
        set_description=set_description
    )

    # 2) Read CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_text = row["test"]
            target_resp = row.get("goal", None)
            # 3) Add each prompt
            mgr.add_prompt_to_set(
                p_set_id,
                prompt_text,
                target_resp
            )

    return p_set_id

def main():
    mgr = ExperimentManager()

    # 1) Create an experiment
    experiment_id = mgr.create_experiment(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        quantization_level="fp16",
        trait_id=4,
        trait_max_or_min="max",
        description="MELBO methodology experiment with 10 shuffles"
    )
    print(f"Created experiment {experiment_id}")

    desired_prompt_set_ids = [1]

    for prompt_set_id in desired_prompt_set_ids:
        mgr.link_experiment_prompt_set(experiment_id, prompt_set_id)

    # 2) Define multiple run parameter dicts
    # This follows the MELBO methodology with 10 different shuffles:
    # - Each shuffle uses a different seed
    # - First 32 instructions of each shuffle used as validation set
    # - Last 100 instructions of each shuffle reserved as test set
    run_params_list = []
    seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    for i, seed in enumerate(seeds):
        rp = {
            "seed": seed,
            "max_new_tokens": 256,
            "target_vector_id": None,
            "num_samples": 10,  # Training on first 12 instructions
            "max_seq_len": 27,
            "source_layer_idx": 10,
            "target_layer_idx": 20,
            "num_factors": 512,
            "forward_batch_size": 1,
            "backward_batch_size": 1,
            "factor_batch_size": 128,
            "num_eval": 128,
            "system_prompt": "You are a person",
            "dim_output_projection": 32,
            "beta": 1.0,
            "max_iters": 10,
            "target_ratio": 0.5,
            "input_scale": None, # if none, gets defined by code
            "run_description": f"MELBO shuffle {i+1}/10 with seed={seed}",
            "shuffle_index": i,    # Using unique index for each shuffle
            "val_size": 4,        # First 32 instructions for validation
            "test_size": 10       # Last 100 instructions for test
        }
        run_params_list.append(rp)

    # 3) Create the runs
    run_ids = mgr.create_multiple_runs(experiment_id, run_params_list)
    print(f"Created {len(run_ids)} runs with MELBO methodology:", run_ids)

    # 4) Execute each run using run_executor
    for i, rid in enumerate(run_ids):
        print(f"Executing run {i+1}/{len(run_ids)} (ID: {rid})...")
        mgr.execute_run(rid)
        print(f"Completed run {i+1}/{len(run_ids)}")

    mgr.close()
    print("All runs completed successfully.")

def add_target_vec():
    mgr = ExperimentManager()

    model, tokenizer = prep_mod_and_tok("meta-llama/Llama-3.2-3B-Instruct")

    target_vector_id = mgr.create_target_vector(
        token_a="I would rate myself a 1",
        token_b="I would rate myself a 5",
        tokenizer=tokenizer,
        model=model, 
        creation_method="diff",
        notes="High-response target vector"
    )

# This is a helper function to allow testing the entry point
def run_main_if_module_is_main():
    if __name__ == "__main__":
        main()

# Call the helper function 
run_main_if_module_is_main()