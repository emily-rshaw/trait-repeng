#!/usr/bin/env python3
"""
Deep Causal Transcoding (DCT) Script

This script implements DCT to modify a language model's behavior by injecting
steering vectors at specific layers.
"""

import os
import math
import random
import click
import torch
import pandas as pd
import dct
import seaborn as sns
from tqdm import tqdm
from torch import vmap
from huggingface_hub import login
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare_dataset(input_path, personality_column="cAGR", answer="y", output_dir=None):
    """Prepare the dataset by filtering based on personality traits."""
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    # Load the dataset
    dataset = pd.read_csv(input_path, encoding="utf-8", encoding_errors='ignore')
    
    # Filter the rows
    filtered = dataset[dataset[personality_column] == answer]
    
    # Create a new DataFrame
    new_df = pd.DataFrame({
        "goal": "Tell me about yourself.",
        "target": filtered["TEXT"]
    })
    
    # Construct an output filename
    output_filename = f"essays_{personality_column}_{answer}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the new dataset to CSV without the index
    new_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(new_df)} rows to {output_path}")
    return output_path


def load_examples(dataset_path, tokenizer, num_samples, system_prompt=None, seed=325):
    """Load and prepare examples from the dataset."""
    random.seed(seed)
    
    # Load the dataset
    dataset = pd.read_csv(dataset_path)
    instructions = dataset['goal'].tolist()
    
    random.shuffle(instructions)
    
    # Prepare chat templates
    if system_prompt is not None:
        chat_init = [{'content': system_prompt, 'role': 'system'}]
    else:
        chat_init = []
    
    # Training examples
    chats = [chat_init + [{'content': content, 'role': 'user'}] 
             for content in instructions[:num_samples]]
    examples = [tokenizer.apply_chat_template(chat, add_special_tokens=False, 
                                           tokenize=False, add_generation_prompt=True) 
                for chat in chats]
    
    # Test examples
    test_chats = [chat_init + [{'content': content, 'role': 'user'}] 
                  for content in instructions[-32:]]
    test_examples = [tokenizer.apply_chat_template(chat, add_special_tokens=False, 
                                                tokenize=False, add_generation_prompt=True) 
                     for chat in test_chats]
    
    return examples, test_examples


def prepare_model_and_tokenizer(model_name, tokenizer_name=None):
    """Load and prepare the model and tokenizer."""
    if tokenizer_name is None:
        tokenizer_name = model_name
        
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, 
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
    """Create a sliced model between the specified layers."""
    # Test the sliced model concept
    test_inputs = tokenizer(["Tell me about yourself"], return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        hidden_states = model(test_inputs["input_ids"], output_hidden_states=True).hidden_states
    
    # Create the actual sliced model
    sliced_model = dct.SlicedModel(
        model, 
        start_layer=source_layer_idx, 
        end_layer=target_layer_idx, 
        layers_name="model.layers"
    )
    
    return sliced_model


def compute_activations(model, tokenizer, sliced_model, examples, source_layer_idx, 
                        max_seq_len, num_samples, forward_batch_size):
    """Compute activations for the training examples."""
    d_model = model.config.hidden_size
    
    X = torch.zeros(num_samples, max_seq_len, d_model, device="cpu")
    Y = torch.zeros(num_samples, max_seq_len, d_model, device="cpu")
    
    for t in tqdm(range(0, num_samples, forward_batch_size)):
        with torch.no_grad():
            model_inputs = tokenizer(
                examples[t:t+forward_batch_size], 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length", 
                max_length=max_seq_len
            ).to(model.device)
            
            hidden_states = model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
            h_source = hidden_states[source_layer_idx]  # b x t x d_model
            unsteered_target = sliced_model(h_source)   # b x t x d_model
            
            X[t:t+forward_batch_size, :, :] = h_source
            Y[t:t+forward_batch_size, :, :] = unsteered_target
    
    return X, Y


def train_dct_model(delta_acts_single, X, Y, num_factors, backward_batch_size, 
                   factor_batch_size, input_scale, dim_output_projection=32, 
                   max_iters=10, beta=1.0):
    """Train the DCT model."""
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


def plot_objective_function(exp_dct, output_dir):
    """Plot and save the objective function values during training."""
    plt.figure()
    plt.plot(exp_dct.objective_values)
    plt.title("Objective Function Values")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.savefig(os.path.join(output_dir, "objective_values.png"))
    plt.close()


def plot_similarity_matrices(U, V, output_dir):
    """Plot and save the similarity matrices for U and V."""
    # U similarity
    with torch.no_grad():
        simu = (U.t() @ U)
        simu = simu[torch.triu(torch.ones_like(simu), diagonal=1).bool()]
    
    plt.figure()
    sns.histplot(simu.cpu())
    plt.title("U Similarity Distribution")
    plt.savefig(os.path.join(output_dir, "u_similarity.png"))
    plt.close()
    
    # V similarity
    with torch.no_grad():
        simv = (V.t() @ V)
        simv = simv[torch.triu(torch.ones_like(simv), diagonal=1).bool()]
    
    plt.figure()
    sns.histplot(simv.cpu())
    plt.title("V Similarity Distribution")
    plt.savefig(os.path.join(output_dir, "v_similarity.png"))
    plt.close()


def rank_vectors(exp_dct, delta_acts_end, X, Y, forward_batch_size, factor_batch_size):
    """Rank the vectors by their effect on the model."""
    scores, indices = exp_dct.rank(
        delta_acts_end, X, Y, 
        target_vec=None,
        batch_size=forward_batch_size, 
        factor_batch_size=factor_batch_size
    )
    
    plt.figure()
    sns.histplot(scores.cpu())
    plt.title("Vector Scores Distribution")
    plt.savefig("vector_scores.png")
    plt.close()
    
    return scores, indices


def evaluate_vectors(model, tokenizer, model_editor, V, indices, input_scale, 
                    source_layer_idx, examples, test_examples, num_eval=5, 
                    max_new_tokens=256, output_dir="outputs"):
    """Evaluate the effect of the top vectors."""
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "steering_results.txt")
    
    with open(results_file, "w") as f:
        # Evaluate unsteered model
        model_editor.restore()
        examples_to_test = examples[:2] + test_examples[:3]  # Select a few examples for testing
        model_inputs = tokenizer(examples_to_test, return_tensors="pt", padding=True).to("cuda")
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        unsteered_completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        f.write("===== UNSTEERED COMPLETIONS =====\n\n")
        for i, completion in enumerate(unsteered_completions):
            f.write(f"Example {i}:\n{completion}\n\n")
        
        # Evaluate top vectors
        for vec_idx in range(min(num_eval, len(indices))):
            vec_id = indices[vec_idx]
            model_editor.restore()
            model_editor.steer(input_scale * V[:, vec_id], source_layer_idx)
            
            model_inputs = tokenizer(examples_to_test, return_tensors="pt", padding=True).to("cuda")
            generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
            steered_completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            f.write(f"===== STEERED BY VECTOR {vec_id} =====\n\n")
            for i, completion in enumerate(steered_completions):
                f.write(f"Example {i}:\n{completion}\n\n")
    
    print(f"Evaluation results saved to {results_file}")


@click.command()
@click.option("--input-path", default="../data/training_datasets/personality/essays.csv", 
              help="Path to the input CSV file")
@click.option("--output-dir", default="../results/raw", help="Directory to save output files")
@click.option("--model-name", default="meta-llama/Llama-3.2-3B-Instruct", 
              help="Name of the model to use")
@click.option("--tokenizer-name", default=None, 
              help="Name of the tokenizer to use (defaults to model-name)")
@click.option("--personality-column", default="cAGR", 
              help="Personality column to filter on")
@click.option("--personality-value", default="y", 
              help="Value of personality column to filter for")
@click.option("--system-prompt", default="You are a helpful assistant", 
              help="System prompt to use")
@click.option("--num-samples", default=1, type=int, 
              help="Number of training samples")
@click.option("--max-seq-len", default=27, type=int, 
              help="Maximum sequence length for training examples")
@click.option("--source-layer-idx", default=10, type=int, 
              help="Source layer index")
@click.option("--target-layer-idx", default=20, type=int, 
              help="Target layer index")
@click.option("--num-factors", default=512, type=int, 
              help="Number of factors to learn")
@click.option("--forward-batch-size", default=1, type=int, 
              help="Batch size for forward passes")
@click.option("--backward-batch-size", default=1, type=int, 
              help="Batch size for backward passes")
@click.option("--factor-batch-size", default=128, type=int, 
              help="Factor batch size")
@click.option("--num-eval", default=128, type=int, 
              help="Number of steering vectors to evaluate")
@click.option("--max-new-tokens", default=256, type=int, 
              help="Maximum number of new tokens to generate")
def main(input_path, output_dir, model_name, tokenizer_name, personality_column, 
         personality_value, system_prompt, num_samples, max_seq_len, source_layer_idx, 
         target_layer_idx, num_factors, forward_batch_size, backward_batch_size, 
         factor_batch_size, num_eval, max_new_tokens):
    """Run the Deep Causal Transcoding (DCT) workflow."""
    # Set up device and random seed
    torch.set_default_device("cuda")
    torch.manual_seed(325)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for HuggingFace token
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HUGGINGFACE_HUB_TOKEN not found. You may not be able to access gated models.")
    
    # Prepare dataset
    dataset_path = prepare_dataset(
        input_path, 
        personality_column=personality_column, 
        answer=personality_value, 
        output_dir=output_dir
    )
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    global model, tokenizer  # Making them global for use in other functions
    model, tokenizer = prepare_model_and_tokenizer(model_name, tokenizer_name)
    
    # Load examples
    print("Preparing examples...")
    examples, test_examples = load_examples(
        dataset_path, 
        tokenizer, 
        num_samples, 
        system_prompt=system_prompt
    )
    
    # Create sliced model
    print("Creating sliced models...")
    sliced_model = create_sliced_model(model, source_layer_idx, target_layer_idx)
    
    # Print model information
    print(f"Model hidden size: {model.config.hidden_size}")
    print(f"Model parameter dtype: {next(model.parameters()).dtype}")
    print(f"Sliced model parameter dtype: {next(sliced_model.parameters()).dtype}")
    
    # Compute activations
    print("Computing activations...")
    X, Y = compute_activations(
        model, 
        tokenizer, 
        sliced_model, 
        examples, 
        source_layer_idx, 
        max_seq_len, 
        num_samples, 
        forward_batch_size
    )
    
    # Set up delta activations
    print("Setting up delta activations...")
    token_idxs = slice(-3, None)  # Target token positions
    delta_acts_single = dct.DeltaActivations(sliced_model, target_position_indices=token_idxs)
    delta_acts = vmap(
        delta_acts_single, 
        in_dims=(1, None, None), 
        out_dims=2,
        chunk_size=factor_batch_size
    )
    
    # Calibrate steering
    print("Calibrating steering...")
    steering_calibrator = dct.SteeringCalibrator(target_ratio=.5)
    input_scale = steering_calibrator.calibrate(
        delta_acts_single, 
        X.cuda(), 
        Y.cuda(), 
        factor_batch_size=factor_batch_size
    )
    print(f"Input scale: {input_scale}")
    
    # Train DCT model
    print("Training DCT model...")
    exp_dct, U, V = train_dct_model(
        delta_acts_single, 
        X, 
        Y, 
        num_factors, 
        backward_batch_size, 
        factor_batch_size, 
        input_scale, 
        dim_output_projection=32, 
        max_iters=10
    )
    
    # Plot and save diagnostic information
    print("Plotting diagnostics...")
    plot_objective_function(exp_dct, output_dir)
    plot_similarity_matrices(U, V, output_dir)
    
    # Create end-to-end sliced model and delta activations
    print("Setting up end-to-end activation...")
    slice_to_end = dct.SlicedModel(
        model, 
        start_layer=source_layer_idx, 
        end_layer=model.config.num_hidden_layers-1, 
        layers_name="model.layers"
    )
    delta_acts_end_single = dct.DeltaActivations(slice_to_end)
    
    # Rank vectors
    print("Ranking vectors...")
    scores, indices = rank_vectors(
        exp_dct, 
        delta_acts_end_single, 
        X, 
        Y, 
        forward_batch_size, 
        factor_batch_size
    )
    
    # Create model editor
    model_editor = dct.ModelEditor(model, layers_name="model.layers")
    
    # Evaluate top vectors
    print("Evaluating top vectors...")
    evaluate_vectors(
        model, 
        tokenizer, 
        model_editor, 
        V, 
        indices, 
        input_scale, 
        source_layer_idx, 
        examples, 
        test_examples, 
        num_eval=num_eval, 
        max_new_tokens=max_new_tokens, 
        output_dir=output_dir
    )
    
    print("DCT workflow completed successfully!")


if __name__ == "__main__":
    main()