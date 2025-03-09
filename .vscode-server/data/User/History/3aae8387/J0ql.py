#!/usr/bin/env python3
"""
Simple UI for the IPIP NEO Personality Trait DCT Modeling tool.

This script provides a user-friendly interface to the main_demo_ipip_mod.py script
for research purposes.
"""

import os
import glob
import sys
import subprocess
import gradio as gr
import pandas as pd
import torch

# Make sure we can import from the same directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import functions from the main script
from main_demo_ipip_mod import list_available_datasets, select_trait_dataset

def get_available_traits(traits_dir="trait_specific"):
    """Get a list of available traits for the dropdown."""
    try:
        # Get list of CSV files in the traits directory
        import glob
        import os
        
        # Get all max_*.csv files (we only need to check one direction as they come in pairs)
        csv_files = glob.glob(os.path.join(traits_dir, "max_*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No trait CSV files found in {traits_dir}")
            
        # Extract trait names from filenames
        trait_names = []
        for file_path in csv_files:
            # Get just the filename without directory
            filename = os.path.basename(file_path)
            # Remove 'max_' prefix and '.csv' suffix
            trait_name = filename[4:-4]
            # Replace hyphens with underscores for consistency
            trait_name = trait_name.replace('-', '_')
            trait_names.append(trait_name)
        
        return sorted(trait_names)
    except Exception as e:
        print(f"Error getting traits: {e}")
        # Return hardcoded defaults from the IPIP NEO model
        return ["a1_trust", "a2_morality", "a3_altruism", "a4_cooperation", "a5_modesty", "a6_sympathy",
                "c1_self_efficacy", "c2_orderliness", "c3_dutifulness", "c4_achievement_striving", 
                "c5_self_discipline", "c6_cautiousness", "e1_friendliness", "e2_gregariousness", 
                "e3_assertiveness", "e4_activity_level", "e5_excitement_seeking", "e6_cheerfulness",
                "n1_anxiety", "n2_anger", "n3_depression", "n4_self_consciousness", 
                "n5_immoderation", "n6_vulnerability", "o1_imagination", "o2_artistic_interests", 
                "o3_emotionality", "o4_adventurousness", "o5_intellect", "o6_liberalism"]

def get_available_models():
    """Return a list of supported models."""
    return [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]

def run_dct_process(trait, direction, model_name, system_prompt, num_samples,
                   max_seq_len, source_layer_idx, target_layer_idx, num_factors, num_eval,
                   traits_dir, output_dir):
    """Run the DCT process and return the results path."""
    # Construct the command with only the essential parameters - use defaults for the rest
    cmd = [
        "python", os.path.join(script_dir, "main_demo_ipip_mod.py"),
        "--trait", trait,
        "--direction", direction,
        "--traits-dir", traits_dir,
        "--output-dir", output_dir
    ]
    
    # Only add non-default parameters if they differ from defaults
    if model_name != "meta-llama/Llama-3.2-3B-Instruct":
        cmd.extend(["--model-name", model_name])
    
    if system_prompt != "You are a person":
        cmd.extend(["--system-prompt", system_prompt])
    
    if num_samples != 4:
        cmd.extend(["--num-samples", str(num_samples)])
    
    if max_seq_len != 27:
        cmd.extend(["--max-seq-len", str(max_seq_len)])
        
    if source_layer_idx != 10:
        cmd.extend(["--source-layer-idx", str(source_layer_idx)])
        
    if target_layer_idx != 20:
        cmd.extend(["--target-layer-idx", str(target_layer_idx)])
        
    if num_factors != 512:
        cmd.extend(["--num-factors", str(num_factors)])
        
    if num_eval != 128:
        cmd.extend(["--num-eval", str(num_eval)])
    
    # Run the process and capture output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Create a log for the output
    log = []
    
    # Stream the output
    for line in process.stdout:
        log.append(line)
        yield "\n".join(log)
    
    # Wait for the process to complete
    process.wait()
    
    # Add any error messages
    for line in process.stderr:
        log.append(f"ERROR: {line}")
        yield "\n".join(log)
    
    # Add the final status
    if process.returncode == 0:
        trait_name = trait.replace("/", "_")
        results_dir = os.path.join(output_dir, f"{direction}_{trait_name}")
        results_file = os.path.join(results_dir, "steering_results.txt")
        
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = f.read()
            
            log.append("\n\nSTEERING RESULTS:")
            log.append(results)
        
        log.append(f"\nSuccess! Results saved to {results_dir}")
    else:
        log.append(f"\nProcess failed with return code {process.returncode}")
    
    yield "\n".join(log)

def view_results(trait, direction, output_dir):
    """View the results for a specific trait."""
    trait_name = trait.replace("/", "_")
    results_dir = os.path.join(output_dir, f"{direction}_{trait_name}")
    results_file = os.path.join(results_dir, "steering_results.txt")
    
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return f.read()
    else:
        return f"No results found for {direction}_{trait}"

def check_gpu_availability():
    """Check if CUDA is available and return GPU info."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        memory_info = [f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB" 
                      for i in range(gpu_count)]
        
        info = f"CUDA available: {gpu_count} GPUs detected\n"
        for i, (name, mem) in enumerate(zip(gpu_names, memory_info)):
            info += f"GPU {i}: {name} with {mem} memory\n"
        return info
    else:
        return "CUDA not available. This application requires a GPU to run."

def create_ui():
    """Create the Gradio UI."""
    # Default paths
    default_traits_dir = "/teamspace/studios/this_studio/data/psychometric_tests/personality/trait_specific"
    default_output_dir = "/teamspace/studios/this_studio/results"
    
    # Ensure output directory exists
    os.makedirs(default_output_dir, exist_ok=True)
    
    # Get traits from the directory or use hardcoded fallback
    available_traits = get_available_traits(default_traits_dir)
    print(f"Found {len(available_traits)} available traits: {available_traits[:5]}...")
    
    # Create the interface
    with gr.Blocks(title="Personality Trait DCT Modeling") as demo:
        gr.Markdown("# Personality Trait DCT Modeling UI")
        gr.Markdown("This tool allows you to use Deep Causal Transcoding to modify language model behavior based on IPIP NEO personality traits.")
        
        # GPU info
        with gr.Accordion("GPU Information", open=False):
            gr.Markdown(check_gpu_availability())
        
        # Settings
        with gr.Tab("Run DCT Model"):
            with gr.Row():
                with gr.Column():
                    trait = gr.Dropdown(
                        choices=available_traits,
                        label="Personality Trait",
                        value=available_traits[0] if available_traits else None,
                        info="Select the personality trait to model"
                    )
                    
                    direction = gr.Radio(
                        choices=["max", "min"],
                        label="Direction",
                        value="max",
                        info="Whether to maximize or minimize the trait"
                    )
                    
                    model_name = gr.Dropdown(
                        choices=get_available_models(),
                        label="Model",
                        value="meta-llama/Llama-3.2-3B-Instruct",
                        info="Select the language model to use"
                    )
                    
                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        value="You are a person",
                        info="System prompt to use for testing"
                    )
                
                with gr.Column():
                    num_samples = gr.Slider(
                        minimum=1,
                        maximum=32,
                        value=4,
                        step=1,
                        label="Number of Training Samples",
                        info="More samples = better results but longer training time"
                    )
                    
                    max_seq_len = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=27,
                        step=1,
                        label="Max Sequence Length",
                        info="Maximum sequence length for training examples"
                    )
                    
                    with gr.Row():
                        source_layer_idx = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=10,
                            step=1,
                            label="Source Layer Index",
                            info="Layer where intervention happens"
                        )
                        
                        target_layer_idx = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=20,
                            step=1,
                            label="Target Layer Index",
                            info="Layer to optimize effect towards"
                        )
                    
                    num_factors = gr.Slider(
                        minimum=16,
                        maximum=1024,
                        value=512,
                        step=16,
                        label="Number of Factors",
                        info="Number of causal factors to learn"
                    )
                    
                    num_eval = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=128,
                        step=1,
                        label="Number of Vectors to Evaluate",
                        info="Number of steering vectors to test"
                    )
            
            with gr.Accordion("Advanced Settings", open=False):
                traits_dir = gr.Textbox(
                    label="Traits Directory",
                    value=default_traits_dir,
                    info="Directory containing trait-specific datasets"
                )
                
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value=default_output_dir,
                    info="Directory to save output files"
                )
            
            run_button = gr.Button("Run DCT Process", variant="primary")
            output_log = gr.Textbox(label="Process Output", lines=20)
            
            run_button.click(
                fn=run_dct_process,
                inputs=[
                    trait, direction, model_name, system_prompt, num_samples,
                    max_seq_len, source_layer_idx, target_layer_idx, num_factors, num_eval,
                    traits_dir, output_dir
                ],
                outputs=output_log
            )
        
        # Results viewer
        with gr.Tab("View Results"):
            with gr.Row():
                view_trait = gr.Dropdown(
                    choices=available_traits,
                    label="Trait",
                    value=available_traits[0] if available_traits else None
                )
                
                view_direction = gr.Radio(
                    choices=["max", "min"],
                    label="Direction",
                    value="max"
                )
                
                view_output_dir = gr.Textbox(
                    label="Output Directory",
                    value=default_output_dir
                )
            
            view_button = gr.Button("View Results")
            results_text = gr.Textbox(label="Results", lines=30)
            
            view_button.click(
                fn=view_results,
                inputs=[view_trait, view_direction, view_output_dir],
                outputs=results_text
            )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False)