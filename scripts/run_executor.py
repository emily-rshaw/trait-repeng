#!/usr/bin/env python3
"""
Run Executor for Deep Causal Transcoding (DCT).

Fetches a run (by run_id) from the database, executes the entire DCT workflow,
and stores steering vectors and outputs back into the database.
"""

import os
import time
import sqlite3
import math
import random
import glob
import torch
import pandas as pd
import dct
import seaborn as sns
from tqdm import tqdm
from torch import vmap
from huggingface_hub import login
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# ------------------------------------------------
# Existing DCT utility functions (lightly modified)
# ------------------------------------------------

import random

def load_examples_for_experiment(
    mgr,
    experiment_id,
    tokenizer,
    num_samples,
    system_prompt=None,
    seed=325
):
    """
    1) Fetch all prompt sets linked to `experiment_id`.
    2) Gather (prompt_id, prompt_text, target_response) from those sets.
    3) Shuffle & do train vs. test split (train = num_samples, test = up to 32).
    4) For each item, apply chat template with system prompt => create final strings
       *BUT* also track which prompt_id each final string corresponds to.
    
    Returns:
      examples, targets, test_examples, test_targets,
      example_prompt_ids, test_prompt_ids
    """
    random.seed(seed)

    # 1) find prompt sets
    ps_rows = mgr.cursor.execute('''
        SELECT ps.prompt_set_id
        FROM prompt_sets ps
        JOIN experiment_prompt_sets eps ON ps.prompt_set_id = eps.prompt_set_id
        WHERE eps.experiment_id = ?
    ''', (experiment_id,)).fetchall()

    if not ps_rows:
        raise ValueError(f"No prompt sets found for experiment_id={experiment_id}")

    all_items = []  # will hold tuples (prompt_id, prompt_text, target_response)
    for row in ps_rows:
        p_set_id = row["prompt_set_id"]
        prompt_rows = mgr.cursor.execute('''
            SELECT prompt_id, prompt_text, target_response
            FROM prompts
            WHERE prompt_set_id = ?
        ''', (p_set_id,)).fetchall()
        for p_row in prompt_rows:
            pid = p_row["prompt_id"]
            txt = p_row["prompt_text"]
            goal = p_row["target_response"]
            all_items.append((pid, txt, goal))

    # shuffle
    random.shuffle(all_items)
    # slice
    clipped = all_items[: num_samples + 32]
    train_data = clipped[: num_samples]
    test_data = clipped[num_samples : num_samples+32]

    chat_init = []
    if system_prompt:
        chat_init = [{'content': system_prompt, 'role': 'system'}]

    # We'll build final strings plus parallel arrays for the prompt IDs
    examples, targets = [], []
    test_examples, test_targets = [], []
    example_prompt_ids, test_prompt_ids = [], []

    # TRAIN
    for (pid, prompt_text, target_resp) in train_data:
        chat = chat_init + [{'content': prompt_text, 'role': 'user'}]
        formatted_chat = tokenizer.apply_chat_template(
            chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True
        )
        examples.append(formatted_chat)
        targets.append(target_resp)
        example_prompt_ids.append(pid)

    # TEST
    for (pid, prompt_text, target_resp) in test_data:
        chat = chat_init + [{'content': prompt_text, 'role': 'user'}]
        formatted_chat = tokenizer.apply_chat_template(
            chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True
        )
        test_examples.append(formatted_chat)
        test_targets.append(target_resp)
        test_prompt_ids.append(pid)

    # Return 6 parallel lists
    return (
        examples, targets,
        test_examples, test_targets,
        example_prompt_ids, test_prompt_ids
    )


def prepare_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        padding_side="left",
        truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda", 
        trust_remote_code=True,
        _attn_implementation="eager"
    )
    return model, tokenizer

def create_sliced_model(model, source_layer_idx, target_layer_idx):
    return dct.SlicedModel(
        model, 
        start_layer=source_layer_idx, 
        end_layer=target_layer_idx, 
        layers_name="model.layers"
    )

def compute_activations(
    model, tokenizer, sliced_model, 
    examples, source_layer_idx, 
    max_seq_len, num_samples, forward_batch_size
):
    d_model = model.config.hidden_size
    X = torch.zeros(num_samples, max_seq_len, d_model, device="cpu")
    Y = torch.zeros(num_samples, max_seq_len, d_model, device="cpu")

    for t in tqdm(range(0, num_samples, forward_batch_size)):
        with torch.no_grad():
            batch_ex = examples[t:t+forward_batch_size]
            model_inputs = tokenizer(
                batch_ex, 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length", 
                max_length=max_seq_len
            ).to(model.device)

            hidden_states = model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
            h_source = hidden_states[source_layer_idx]
            unsteered_target = sliced_model(h_source)
            X[t:t+forward_batch_size, :, :] = h_source.cpu()
            Y[t:t+forward_batch_size, :, :] = unsteered_target.cpu()
    return X, Y

def train_dct_model(
    delta_acts_single, X, Y, 
    num_factors, backward_batch_size, factor_batch_size, 
    input_scale, dim_output_projection, 
    max_iters, beta
):
    exp_dct = dct.ExponentialDCT(num_factors=num_factors)
    U, V = exp_dct.fit(
        delta_acts_single, X, Y, 
        batch_size=backward_batch_size,
        factor_batch_size=factor_batch_size,
        init="jacobian",
        d_proj=dim_output_projection,
        input_scale=input_scale,
        max_iters=max_iters,
        beta=beta
    )
    return exp_dct, U, V

def rank_vectors(exp_dct, delta_acts_end_single, X, Y, forward_batch_size, factor_batch_size):
    scores, indices = exp_dct.rank(
        delta_acts_end_single, X, Y,
        target_vec=None,
        batch_size=forward_batch_size,
        factor_batch_size=factor_batch_size
    )
    return scores, indices

def evaluate_vectors_and_capture(
    model, tokenizer, model_editor, V, indices, input_scale,
    source_layer_idx, 
    examples, test_examples, 
    targets, test_targets,
    example_prompt_ids, test_prompt_ids,  # new parallel lists
    num_eval, max_new_tokens
):
    """
    Similar to old evaluate_vectors_and_capture, but we also track the prompt_id
    for each example/test item, storing them in 'steered' and 'unsteered' dict.
    
    Returns:
      {
        "unsteered": [(prompt_id, prompt_text, completion_text), ...],
        "steered": {
           vec_idx_1: [(prompt_id, prompt_text, completion_text), ...],
           ...
        }
      }
    """
    results = {"unsteered": [], "steered": {}}

    model_editor.restore()
    # for demonstration, we only evaluate 2 from train + 3 from test
    examples_to_test = examples[:2] + test_examples[:3]
    prompt_ids_to_test = example_prompt_ids[:2] + test_prompt_ids[:3]

    model_inputs = tokenizer(examples_to_test, return_tensors="pt", padding=True).to("cuda")
    unsteered_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    unsteered_completions = tokenizer.batch_decode(unsteered_ids, skip_special_tokens=True)

    # store unsteered
    for (pid, prompt_str, completion_str) in zip(prompt_ids_to_test, examples_to_test, unsteered_completions):
        results["unsteered"].append((pid, prompt_str, completion_str))

    total_vecs = min(num_eval, len(indices))
    for i in tqdm(range(total_vecs), desc="Evaluating steering vectors"):
        vec_idx = indices[i]

        model_editor.restore()
        model_editor.steer(input_scale * V[:, vec_idx], source_layer_idx)

        steered_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        steered_completions = tokenizer.batch_decode(steered_ids, skip_special_tokens=True)

        # store with prompt_id
        gathered = []
        for (pid, prompt_str, completion_str) in zip(prompt_ids_to_test, examples_to_test, steered_completions):
            gathered.append((pid, prompt_str, completion_str))

        results["steered"][vec_idx] = gathered

    return results

def extract_after_assistant(full_text: str) -> str:
    """
    Returns everything after the substring 'assistant'.
    If 'assistant' is not found, returns the full text unchanged.
    """
    # Lowercase search for robust matching
    idx = full_text.lower().find("assistant")
    if idx == -1:
        return full_text  # fallback: 'assistant' not found
    # Get everything after 'assistant'
    # If your text is "assistant:" or "assistant\n", adjust accordingly
    # e.g. if you want to skip "assistant" plus a space or colon, do + 9 or so
    return full_text[idx + len("assistant"):].lstrip(": \n")


def extract_after_assistant(full_text: str) -> str:
    """
    Strips everything up to 'assistant' if desired.
    """
    idx = full_text.lower().find("assistant")
    if idx == -1:
        return full_text
    return full_text[idx + len("assistant"):].lstrip(": \n")


def execute_run(run_id, db_path="results/database/experiments.db"):
    """
    1) Fetch run + experiment from DB
    2) load examples from DB-based approach (with prompt IDs)
    3) do DCT pipeline
    4) store vectors + outputs in DB (including prompt_id)
    5) update run
    6) return final info
    """
    start_time = time.time()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # fetch run
    row = cursor.execute("""
        SELECT r.*, e.*
        FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE r.run_id = ?
    """, (run_id,)).fetchone()

    if not row:
        conn.close()
        raise ValueError(f"No run found with run_id={run_id}")

    # gather run params
    experiment_id = row["experiment_id"]
    model_name = row["model_name"]
    seed = row["seed"]
    max_new_tokens = row["max_new_tokens"]
    num_samples = row["num_samples"]
    max_seq_len = row["max_seq_len"]
    source_layer_idx = row["source_layer_idx"]
    target_layer_idx = row["target_layer_idx"]
    num_factors = row["num_factors"]
    forward_batch_size = row["forward_batch_size"]
    backward_batch_size = row["backward_batch_size"]
    factor_batch_size = row["factor_batch_size"]
    num_eval = row["num_eval"]
    system_prompt = row["system_prompt"]
    dim_output_projection = row["dim_output_projection"]
    beta = row["beta"]
    max_iters = row["max_iters"]
    target_ratio = row["target_ratio"]
    input_scale_db = row["input_scale"]
    run_description = row["run_description"]

    torch.set_default_device("cuda")
    torch.manual_seed(seed)

    from huggingface_hub import login
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
    if hf_token:
        login(token=hf_token)

    model, tokenizer = prepare_model_and_tokenizer(model_name)

    # connect to the manager to load examples
    from experiment_manager import ExperimentManager
    mgr = ExperimentManager(db_path)

    # load DB-based examples with prompt IDs
    (
        examples, targets, 
        test_examples, test_targets,
        example_prompt_ids, test_prompt_ids
    ) = load_examples_for_experiment(
        mgr,
        experiment_id,
        tokenizer,  # We'll set 'tokenizer' after we load the actual HF model
        num_samples,
        system_prompt=system_prompt,
        seed=seed
    )

    mgr.close()

    # create sliced model
    from dct import SlicedModel, DeltaActivations
    sliced_model = create_sliced_model(model, source_layer_idx, target_layer_idx)

    # compute activations
    eff_num_samples = min(num_samples, len(examples))
    X, Y = compute_activations(
        model, tokenizer, sliced_model,
        examples, source_layer_idx,
        max_seq_len, eff_num_samples,
        forward_batch_size
    )

    # delta acts
    delta_acts_single = dct.DeltaActivations(sliced_model, target_position_indices=slice(-3,None))
    _ = vmap(delta_acts_single, in_dims=(1,None,None), out_dims=2, chunk_size=factor_batch_size)

    # calibrate steering
    steering_calibrator = dct.SteeringCalibrator(target_ratio=target_ratio)
    if input_scale_db is not None and input_scale_db > 0:
        input_scale = input_scale_db
    else:
        input_scale = steering_calibrator.calibrate(
            delta_acts_single, 
            X.cuda(), 
            Y.cuda(), 
            factor_batch_size=factor_batch_size
        )

    # train DCT
    exp_dct, U, V = train_dct_model(
        delta_acts_single, X, Y,
        num_factors, backward_batch_size, factor_batch_size,
        input_scale, dim_output_projection,
        max_iters, beta
    )

    # rank
    slice_to_end = dct.SlicedModel(
        model,
        start_layer=source_layer_idx,
        end_layer=model.config.num_hidden_layers - 1,
        layers_name="model.layers"
    )
    delta_acts_end_single = dct.DeltaActivations(slice_to_end)
    scores, raw_indices = rank_vectors(exp_dct, delta_acts_end_single, X, Y, forward_batch_size, factor_batch_size)
    indices = raw_indices.cpu().int().tolist()

    # evaluate + capture
    from dct import ModelEditor
    model_editor = ModelEditor(model, layers_name="model.layers")
    eval_results = evaluate_vectors_and_capture(
        model, tokenizer, model_editor, V, indices,
        input_scale, source_layer_idx,
        examples, test_examples,
        targets, test_targets,
        example_prompt_ids, test_prompt_ids,  # new parallel IDs
        num_eval, max_new_tokens
    )

    # store vectors, outputs
    # steering_vectors
    for i, vec_idx in enumerate(indices):
        vec_blob = V[:, vec_idx].detach().cpu().numpy().tobytes()
        rank_score = float(scores[i]) if i < len(scores) else None
        cursor.execute("""
            INSERT INTO steering_vectors (
                created_by_run_id,
                vector_rank_score,
                vector_rank_index,
                vector_data,
                is_random
            ) VALUES (?, ?, ?, ?, ?)
        """, (run_id, rank_score, vec_idx, vec_blob, 0))
        vector_id = cursor.lastrowid

        # for steered completions
        if vec_idx in eval_results["steered"]:
            for (prompt_id, prompt_str, completion_str) in eval_results["steered"][vec_idx]:
                final_text = extract_after_assistant(completion_str)
                cursor.execute("""
                    INSERT INTO outputs (
                        run_id,
                        prompt_id,
                        vector_id,
                        output_text
                    ) VALUES (?, ?, ?, ?)
                """, (run_id, prompt_id, vector_id, final_text))

    # unsteered
    for (prompt_id, prompt_str, completion_str) in eval_results["unsteered"]:
        final_text = extract_after_assistant(completion_str)
        cursor.execute("""
            INSERT INTO outputs (
                run_id,
                prompt_id,
                vector_id,
                output_text
            ) VALUES (?, ?, ?, ?)
        """, (run_id, prompt_id, None, final_text))

    # update run
    duration = time.time() - start_time
    cursor.execute("""
        UPDATE runs
        SET duration_in_seconds = ?,
            input_scale = ?
        WHERE run_id = ?
    """, (duration, input_scale, run_id))

    conn.commit()
    conn.close()

    return {
        "run_id": run_id,
        "duration_in_seconds": duration,
        "input_scale": input_scale,
        "num_vectors_stored": len(indices),
        "run_description": run_description
    }