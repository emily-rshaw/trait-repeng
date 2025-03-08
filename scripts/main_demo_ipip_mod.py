#!/usr/bin/env python3
"""
Deep Causal Transcoding (DCT) Script for IPIP NEO Personality Traits

This script implements DCT to modify a language model's behavior by injecting
steering vectors at specific layers, adapted to work with IPIP NEO personality datasets.
"""

import os
import math
import random
import glob
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


def list_available_datasets(traits_dir="trait_specific"):
    """List all available trait-specific datasets in the directory."""
    datasets = glob.glob(os.path.join(traits_dir, "*.csv"))
    
    # Group datasets by trait
    trait_groups = {}
    for dataset in datasets:
        filename = os.path.basename(dataset)
        trait_name = filename.split('_', 1)[1].rsplit('.', 1)[0]  # Extract trait name from filename
        direction = "maximize" if filename.startswith("max_") else "minimize"
        
        if trait_name not in trait_groups:
            trait_groups[trait_name] = []
        trait_groups[trait_name].append((direction, dataset))
    
    # Print available datasets
    print("Available trait datasets:")
    for trait, dataset_list in trait_groups.items():
        print(f"  {trait}:")
        for direction, path in dataset_list:
            print(f"    - {direction}: {os.path.basename(path)}")
    
    return trait_groups


def select_trait_dataset(traits_dir="trait_specific", trait=None, direction="max"):
    """Select a dataset based on trait and direction (max/min)."""
    if not os.path.exists(traits_dir):
        raise ValueError(f"Traits directory '{traits_dir}' does not exist")
    
    # Get all datasets
    datasets = glob.glob(os.path.join(traits_dir, f"{direction}_*.csv"))
    
    if not datasets:
        raise ValueError(f"No {direction} datasets found in {traits_dir}")
    
    # If trait is specified, find matching dataset
    if trait:
        # Files are using the same format as provided in traits (no conversion needed)
        # First try exact match
        matching_datasets = [d for d in datasets if f"{direction}_{trait}.csv" == os.path.basename(d)]
        
        # If no exact match, try case-insensitive partial match
        if not matching_datasets:
            matching_datasets = [d for d in datasets if trait.lower() in os.path.basename(d).lower()]
            
        if not matching_datasets:
            raise ValueError(f"No dataset found for trait '{trait}' with direction '{direction}'")
        
        return matching_datasets[0]
    
    # Otherwise, return the first dataset
    return datasets[0]


def load_examples(dataset_path, tokenizer, num_samples, system_prompt=None, seed=325):
    """Load and prepare examples from the IPIP NEO dataset."""
    random.seed(seed)
    
    # Load the dataset
    dataset = pd.read_csv(dataset_path)
    
    # Check if dataset has the expected format
    if 'test' not in dataset.columns or 'goal' not in dataset.columns:
        raise ValueError(f"Dataset {dataset_path} doesn't have the expected 'test' and 'goal' columns")
    
    # Get test questions and goals
    questions = dataset['test'].tolist()
    goals = dataset['goal'].tolist()
    
    # Combine questions and their expected responses
    paired_data = list(zip(questions, goals))
    random.shuffle(paired_data)
    paired_data = paired_data[:num_samples + 32]  # Take enough for training and testing
    
    # Prepare chat templates
    if system_prompt is not None:
        chat_init = [{'content': system_prompt, 'role': 'system'}]
    else:
        chat_init = []
    
    # Format the examples with chat template
    examples = []
    targets = []
    
    for i in range(min(num_samples, len(paired_data))):
        question, goal = paired_data[i]
        
        # Create chat with user query
        chat = chat_init + [{'content': question, 'role': 'user'}]
        formatted_chat = tokenizer.apply_chat_template(chat, add_special_tokens=False, 
                                                     tokenize=False, add_generation_prompt=True)
        examples.append(formatted_chat)
        targets.append(goal)
    
    # Test examples
    test_examples = []
    test_targets = []
    
    for i in range(num_samples, min(num_samples + 32, len(paired_data))):
        question, goal = paired_data[i]
        
        # Create chat with user query
        chat = chat_init + [{'content': question, 'role': 'user'}]
        formatted_chat = tokenizer.apply_chat_template(chat, add_special_tokens=False, 
                                                     tokenize=False, add_generation_prompt=True)
        test_examples.append(formatted_chat)
        test_targets.append(goal)
    
    return examples, targets, test_examples, test_targets


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
                    source_layer_idx, examples, test_examples, targets, test_targets,
                    num_eval=5, max_new_tokens=256, output_dir="outputs"):
    """Evaluate the effect of the top vectors."""
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "steering_results.txt")
    
    with open(results_file, "w") as f:
        trait_name = os.path.basename(output_dir)
        f.write(f"# Personality Trait Vector Analysis: {trait_name}\n")
        f.write("This file contains analysis of different steering vectors and their impact on model responses.\n\n")
        
        # Evaluate unsteered model
        model_editor.restore()
        examples_to_test = examples[:2] + test_examples[:3]  # Select a few examples for testing
        targets_to_compare = targets[:2] + test_targets[:3]
        
        model_inputs = tokenizer(examples_to_test, return_tensors="pt", padding=True).to("cuda")
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        unsteered_completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        f.write("## BASELINE (UNSTEERED MODEL)\n")
        f.write("These responses show the model's behavior without any personality trait steering.\n\n")
        
        for i, (completion, target) in enumerate(zip(unsteered_completions, targets_to_compare)):
            # Extract question and response more safely
            try:
                if 'user:' in completion.lower() and 'assistant:' in completion.lower():
                    # Try to extract between user: and assistant: (case insensitive)
                    parts = completion.lower().split('user:')
                    if len(parts) > 1:
                        user_part = parts[1]
                        parts = user_part.split('assistant:')
                        question = parts[0].strip()
                        response = parts[1].strip() if len(parts) > 1 else "Failed to extract response"
                else:
                    # Try to find the question within the completion
                    original_example = examples_to_test[i]
                    question_part = original_example.split('<|assistant|>', 1)[0].strip()
                    if '<|user|>' in question_part:
                        question = question_part.split('<|user|>', 1)[1].strip()
                    else:
                        question = "Could not extract question"
                    
                    # Get everything after the assistant marker
                    if '<|assistant|>' in completion:
                        response = completion.split('<|assistant|>', 1)[1].strip()
                    else:
                        response = completion
            except Exception as e:
                question = "Error extracting question"
                response = completion
            
            f.write(f"### Example {i+1}\n")
            f.write(f"**Question:** {question}\n")
            f.write(f"**Target Score:** {target}\n")
            f.write(f"**Model Response:**\n\n{response}\n\n")
            f.write("---\n\n")
        
        # Evaluate top vectors
        for vec_idx in range(min(num_eval, len(indices))):
            vec_id = indices[vec_idx]
            model_editor.restore()
            model_editor.steer(input_scale * V[:, vec_id], source_layer_idx)
            
            model_inputs = tokenizer(examples_to_test, return_tensors="pt", padding=True).to("cuda")
            generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
            steered_completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            f.write(f"## VECTOR {vec_id} ANALYSIS\n")
            f.write(f"This section shows responses when steering with vector {vec_id}.\n\n")
            
            for i, (completion, target) in enumerate(zip(steered_completions, targets_to_compare)):
                # Extract question and response more safely
                try:
                    if 'user:' in completion.lower() and 'assistant:' in completion.lower():
                        # Try to extract between user: and assistant: (case insensitive)
                        parts = completion.lower().split('user:')
                        if len(parts) > 1:
                            user_part = parts[1]
                            parts = user_part.split('assistant:')
                            question = parts[0].strip()
                            response = parts[1].strip() if len(parts) > 1 else "Failed to extract response"
                    else:
                        # Try to find the question within the completion
                        original_example = examples_to_test[i]
                        question_part = original_example.split('<|assistant|>', 1)[0].strip()
                        if '<|user|>' in question_part:
                            question = question_part.split('<|user|>', 1)[1].strip()
                        else:
                            question = "Could not extract question"
                        
                        # Get everything after the assistant marker
                        if '<|assistant|>' in completion:
                            response = completion.split('<|assistant|>', 1)[1].strip()
                        else:
                            response = completion
                except Exception as e:
                    question = "Error extracting question"
                    response = completion
                
                f.write(f"### Example {i+1}\n")
                f.write(f"**Question:** {question}\n")
                f.write(f"**Target Score:** {target}\n")
                f.write(f"**Model Response:**\n\n{response}\n\n")
                f.write("---\n\n")
    
    print(f"Evaluation results saved to {results_file}")


@click.command()
@click.option("--traits-dir", default="/teamspace/studios/this_studio/data/psychometric_tests/personality/trait_specific", help="Directory containing trait-specific datasets")
@click.option("--trait", default=None, help="Trait to maximize/minimize (e.g., 'extraversion', 'n1_anxiety')")
@click.option("--direction", default="max", type=click.Choice(['max', 'min']), 
              help="Whether to maximize or minimize the trait")
@click.option("--output-dir", default="./results", help="Directory to save output files")
@click.option("--model-name", default="meta-llama/Llama-3.2-3B-Instruct", 
              help="Name of the model to use")
@click.option("--tokenizer-name", default=None, 
              help="Name of the tokenizer to use (defaults to model-name)")
@click.option("--system-prompt", default="You are a person", 
              help="System prompt to use")
@click.option("--num-samples", default=4, type=int, 
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
@click.option("--list-datasets", is_flag=True, help="List available datasets and exit")
def main(traits_dir, trait, direction, output_dir, model_name, tokenizer_name, system_prompt,
         num_samples, max_seq_len, source_layer_idx, target_layer_idx, num_factors,
         forward_batch_size, backward_batch_size, factor_batch_size, num_eval,
         max_new_tokens, list_datasets):
    """Run the Deep Causal Transcoding (DCT) workflow with IPIP NEO datasets."""
    # List available datasets if requested
    trait_groups = list_available_datasets(traits_dir)
    if list_datasets:
        return
    
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
    
    # Select dataset based on trait and direction
    dataset_path = select_trait_dataset(traits_dir, trait, direction)
    trait_name = os.path.basename(dataset_path).split('.')[0]
    
    print(f"Using dataset: {dataset_path}")
    
    # Create trait-specific output directory
    trait_output_dir = os.path.join(output_dir, trait_name)
    os.makedirs(trait_output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    global model, tokenizer  # Making them global for use in other functions
    model, tokenizer = prepare_model_and_tokenizer(model_name, tokenizer_name)
    
    # Load examples
    print("Preparing examples...")
    examples, targets, test_examples, test_targets = load_examples(
        dataset_path, 
        tokenizer, 
        num_samples, 
        system_prompt=system_prompt
    )
    
    print(f"Loaded {len(examples)} training examples and {len(test_examples)} test examples")
    
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
        min(num_samples, len(examples)), 
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
    plot_objective_function(exp_dct, trait_output_dir)
    plot_similarity_matrices(U, V, trait_output_dir)
    
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
        targets,
        test_targets,
        num_eval=num_eval, 
        max_new_tokens=max_new_tokens, 
        output_dir=trait_output_dir
    )
    
    # Save trained vectors
    vectors_path = os.path.join(trait_output_dir, "steering_vectors.pt")
    torch.save({
        "V": V,
        "indices": indices,
        "scores": scores,
        "input_scale": input_scale,
        "source_layer_idx": source_layer_idx,
        "trait": trait_name
    }, vectors_path)
    print(f"Saved steering vectors to {vectors_path}")
    
    print(f"DCT workflow completed successfully for trait: {trait_name}!")


if __name__ == "__main__":
    main()