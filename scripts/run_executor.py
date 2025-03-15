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

def select_trait_dataset(traits_dir, trait_name, direction):
    """
    Example usage:
      select_trait_dataset('trait_specific', 'extraversion', 'max')
    """
    if not os.path.exists(traits_dir):
        raise ValueError(f"Traits directory '{traits_dir}' does not exist")

    pattern = f"{direction}_{trait_name}.csv"
    candidates = glob.glob(os.path.join(traits_dir, pattern))
    if not candidates:
        # fallback: try partial match or raise error if your design demands exact files
        raise ValueError(f"No dataset found for trait '{trait_name}' with direction '{direction}' in {traits_dir}")

    return candidates[0]

def load_examples(dataset_path, tokenizer, num_samples, system_prompt, seed):
    random.seed(seed)
    dataset = pd.read_csv(dataset_path)
    if 'test' not in dataset.columns or 'goal' not in dataset.columns:
        raise ValueError(f"Dataset {dataset_path} missing 'test'/'goal' columns")

    questions = dataset['test'].tolist()
    goals = dataset['goal'].tolist()
    paired_data = list(zip(questions, goals))
    random.shuffle(paired_data)
    paired_data = paired_data[: num_samples + 32]

    chat_init = []
    if system_prompt:
        chat_init = [{'content': system_prompt, 'role': 'system'}]

    examples, targets = [], []
    for i in range(min(num_samples, len(paired_data))):
        question, goal = paired_data[i]
        chat = chat_init + [{'content': question, 'role': 'user'}]
        formatted_chat = tokenizer.apply_chat_template(
            chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True
        )
        examples.append(formatted_chat)
        targets.append(goal)

    test_examples, test_targets = [], []
    for i in range(num_samples, min(num_samples + 32, len(paired_data))):
        question, goal = paired_data[i]
        chat = chat_init + [{'content': question, 'role': 'user'}]
        formatted_chat = tokenizer.apply_chat_template(
            chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True
        )
        test_examples.append(formatted_chat)
        test_targets.append(goal)

    return examples, targets, test_examples, test_targets

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
    source_layer_idx, examples, test_examples, 
    targets, test_targets, num_eval, max_new_tokens
):
    """
    Evaluate the effect of the top vectors but return completions
    so we can store them in DB.
    Returns a dict with:
      {
        "unsteered": [(prompt_str, completion_str), ...],
        "steered": {
           vec_id_1: [(prompt_str, completion_str), ...],
           vec_id_2: ...
        }
      }
    """
    results = {"unsteered": [], "steered": {}}

    model_editor.restore()
    # Currently: small subset for demonstration
    # If you want to evaluate the entire dataset, do:
    # examples_to_test = examples + test_examples
    examples_to_test = examples[:2] + test_examples[:3]

    model_inputs = tokenizer(examples_to_test, return_tensors="pt", padding=True).to("cuda")
    unsteered_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    unsteered_completions = tokenizer.batch_decode(unsteered_ids, skip_special_tokens=True)

    # Store unsteered
    for (prompt_text, completion_text) in zip(examples_to_test, unsteered_completions):
        results["unsteered"].append((prompt_text, completion_text))

    # Evaluate top vectors with progress bar
    total_vecs = min(num_eval, len(indices))
    for i in tqdm(range(total_vecs), desc="Evaluating steering vectors"):
        # Here, indices[i] is already an int after we fix the code below
        vec_idx = indices[i]

        # restore
        model_editor.restore()
        # steer with the integer index
        model_editor.steer(input_scale * V[:, vec_idx], source_layer_idx)

        # generate
        steered_ids = model.generate(
            **model_inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False
        )
        steered_completions = tokenizer.batch_decode(steered_ids, skip_special_tokens=True)

        # store
        results["steered"][vec_idx] = []
        for (prompt_text, completion_text) in zip(examples_to_test, steered_completions):
            results["steered"][vec_idx].append((prompt_text, completion_text))

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


def execute_run(run_id, db_path="results/database/experiments.db", 
                traits_dir="/teamspace/studios/this_studio/data/psychometric_tests/personality/trait_specific"):
    """
    1) Fetch run (and experiment) data from DB.
    2) Run the DCT pipeline.
    3) Store vectors & outputs in DB (including steered outputs).
    4) Update the run row.
    5) Return final info.
    """
    start_time = time.time()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1) Load run + experiment
    row = cursor.execute("""
        SELECT 
            r.run_id,
            r.experiment_id,
            r.target_vector_id,
            r.seed,
            r.max_new_tokens,
            r.num_samples,
            r.max_seq_len,
            r.source_layer_idx,
            r.target_layer_idx,
            r.num_factors,
            r.forward_batch_size,
            r.backward_batch_size,
            r.factor_batch_size,
            r.num_eval,
            r.system_prompt,
            r.dim_output_projection,
            r.beta,
            r.max_iters,
            r.target_ratio,
            r.input_scale,
            r.run_description,
            e.model_name,
            e.quantization_level,
            e.trait_id,
            e.trait_max_or_min
        FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE r.run_id = ?
    """, (run_id,)).fetchone()

    if not row:
        conn.close()
        raise ValueError(f"No run found with run_id={run_id}")

    # Possibly fetch trait_name from traits
    trait_id = row["trait_id"]
    if trait_id is None:
        trait_name = "unknown"
    else:
        trait_row = cursor.execute("""
            SELECT trait_name
            FROM traits
            WHERE trait_id = ?
        """, (trait_id,)).fetchone()
        trait_name = trait_row["trait_name"] if trait_row else "unknown"

    direction = row["trait_max_or_min"]
    if direction not in ("max", "min"):
        raise ValueError(f"trait_max_or_min must be 'max' or 'min'. Got {direction}")

    # 2) Extract run parameters
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

    # Set seeds
    torch.set_default_device("cuda")
    torch.manual_seed(seed)

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
    if hf_token:
        login(token=hf_token)

    # 3) Hardcode dataset path for now
    dataset_path = "/teamspace/studios/this_studio/data/psychometric_tests/personality/trait_specific/max_a1_trust.csv"

    # 4) Prepare model & tokenizer
    model, tokenizer = prepare_model_and_tokenizer(model_name)

    # 5) Load examples
    examples, targets, test_examples, test_targets = load_examples(
        dataset_path, tokenizer, num_samples, system_prompt, seed
    )

    # 6) Create sliced model
    sliced_model = create_sliced_model(model, source_layer_idx, target_layer_idx)

    # 7) Compute activations
    eff_num_samples = min(num_samples, len(examples))
    X, Y = compute_activations(
        model, tokenizer, sliced_model,
        examples, source_layer_idx,
        max_seq_len, eff_num_samples,
        forward_batch_size
    )

    # 8) DeltaActivations
    token_idxs = slice(-3, None)
    delta_acts_single = dct.DeltaActivations(sliced_model, target_position_indices=token_idxs)
    _ = vmap(delta_acts_single, in_dims=(1, None, None), out_dims=2, chunk_size=factor_batch_size)

    # 9) Possibly calibrate input_scale
    steering_calibrator = dct.SteeringCalibrator(target_ratio=target_ratio)
    if input_scale_db is not None and input_scale_db > 0:
        input_scale = input_scale_db
    else:
        input_scale = steering_calibrator.calibrate(
            delta_acts_single, X.cuda(), Y.cuda(),
            factor_batch_size=factor_batch_size
        )

    # 10) Train DCT
    exp_dct, U, V = train_dct_model(
        delta_acts_single, X, Y,
        num_factors, backward_batch_size, factor_batch_size,
        input_scale, dim_output_projection,
        max_iters, beta
    )

    # 11) rank
    slice_to_end = dct.SlicedModel(
        model,
        start_layer=source_layer_idx,
        end_layer=model.config.num_hidden_layers - 1,
        layers_name="model.layers"
    )
    delta_acts_end_single = dct.DeltaActivations(slice_to_end)
    scores, raw_indices = rank_vectors(exp_dct, delta_acts_end_single, X, Y, forward_batch_size, factor_batch_size)

    # Convert raw_indices (torch tensor) -> list of ints
    indices = raw_indices.cpu().int().tolist()

    # 12) Evaluate
    model_editor = dct.ModelEditor(model, layers_name="model.layers")
    eval_results = evaluate_vectors_and_capture(
        model, tokenizer, model_editor, V, indices,
        input_scale, source_layer_idx,
        examples, test_examples,
        targets, test_targets,
        num_eval, max_new_tokens
    )

    # 13) Store vectors & outputs in DB
    for i, vec_idx in enumerate(indices):
        # vec_idx is now a python int
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

        # Insert steered completions if this vec_idx is in eval_results["steered"]
        if vec_idx in eval_results["steered"]:
            for (prompt_text, completion_text) in eval_results["steered"][vec_idx]:
                final_text = extract_after_assistant(completion_text)
                cursor.execute("""
                    INSERT INTO outputs (
                        run_id,
                        prompt_id,
                        vector_id,
                        output_text
                    ) VALUES (?, ?, ?, ?)
                """, (run_id, None, vector_id, final_text))

    # Unsteered completions
    for (prompt_text, completion_text) in eval_results["unsteered"]:
        final_text = extract_after_assistant(completion_text)
        cursor.execute("""
            INSERT INTO outputs (
                run_id,
                prompt_id,
                vector_id,
                output_text
            ) VALUES (?, ?, ?, ?)
        """, (run_id, None, None, final_text))

    # 14) Update run
    duration = time.time() - start_time
    cursor.execute("""
        UPDATE runs
        SET duration_in_seconds = ?,
            input_scale = ?
        WHERE run_id = ?
    """, (duration, input_scale, run_id))

    conn.commit()
    conn.close()

    # 15) Return final info
    return {
        "run_id": run_id,
        "trait_name": trait_name,
        "direction": direction,
        "duration_in_seconds": duration,
        "input_scale": input_scale,
        "num_vectors_stored": len(indices),
        "run_description": run_description
    }