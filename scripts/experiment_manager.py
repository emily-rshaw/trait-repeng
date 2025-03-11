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

# 1) Import the run_executor
from run_executor import execute_run as run_dct

DB_PATH = "results/database/experiments.db"


class ExperimentManager:
    def __init__(self, db_path=DB_PATH):
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

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
        """
        Insert a run row into 'runs', passing all columns explicitly via run_params.
        *Do not* store duration_in_seconds here, because run_executor will handle it.
        """
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
            run_description
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def main():
    mgr = ExperimentManager()

    # 1) Create an experiment
    experiment_id = mgr.create_experiment(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        quantization_level="fp16",
        trait_id=None,
        trait_max_or_min="max",
        description="Demo experiment with no hardcoded run params, integrated with run_executor"
    )
    print(f"Created experiment {experiment_id}")

    # 2) Define multiple run parameter dicts
    run_params_list = []
    for seed in [101, 102]:
        rp = {
            "seed": seed,
            "max_new_tokens": 256,
            "num_samples": 4,
            "max_seq_len": 27,
            "source_layer_idx": 10,
            "target_layer_idx": 20,
            "num_factors": 512,
            "forward_batch_size": 1,
            "backward_batch_size": 1,
            "factor_batch_size": 128,
            "num_eval": 128,
            "system_prompt": None,
            "dim_output_projection": 32,
            "beta": 1.0,
            "max_iters": 10,
            "target_ratio": 0.5,
            "input_scale": None, # if none, gets defined by code
            "run_description": f"Demo run with seed={seed}"
        }
        run_params_list.append(rp)

    # 3) Create the runs
    run_ids = mgr.create_multiple_runs(experiment_id, run_params_list)
    print("Created runs:", run_ids)

    # 4) Execute each run using run_executor
    for rid in run_ids:
        mgr.execute_run(rid)

    mgr.close()

if __name__ == "__main__":
    main()
