# Cell
import dct as dct 
from tqdm import tqdm
import math
from torch import vmap
import torch
torch.set_default_device("cuda")
torch.manual_seed(325)


# Cell
import pandas as pd
import os

# Path to the original CSV file
input_path = "../tests/training_datasets/personality/essays.csv"

# Load the dataset
dataset = pd.read_csv(input_path, encoding = "utf-8", encoding_errors='ignore')

# Define the personality columns to iterate over
personality_columns = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]

# Directory to save new CSV files (same as the input file location)
output_dir = os.path.dirname(input_path)

# For each personality column and for each possible value ("y" and "n"),
# filter the dataset and create a new CSV file.
for col in personality_columns:
    for answer in ["y"]:
        # Filter the rows where the personality column equals the answer
        filtered = dataset[dataset[col] == answer]
        
        # Create a new DataFrame with columns "goal" and "target"
        new_df = pd.DataFrame({
            "goal": "Tell me about yourself.",
            "target": filtered["TEXT"]
        })
        
        # Construct an output filename (e.g., essays_cEXT_y.csv)
        output_filename = f"essays_{col}_{answer}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the new dataset to CSV without the index
        new_df.to_csv(output_path, index=False)
        
        print(f"Saved {len(new_df)} rows to {output_path}")


# Cell
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
TOKENIZER_NAME = MODEL_NAME

INPUT_SCALE = None          # norm of steering vectors; set to None to use default calibration

NUM_SAMPLES = 1             # number of training samples
FORWARD_BATCH_SIZE = 1      # batch size to use when computing forward passes
BACKWARD_BATCH_SIZE = 1     # batch size to use when computing backward passes (usually smaller)
MAX_SEQ_LEN = 27            # max length for truncating training examples

CALIBRATION_SAMPLE_SIZE = 30        # sample size for random directions used to calibrate input/output scales
CALIBRATION_PROMPT_SAMPLE_SIZE = 1  # prompt sample size for calibrating input scale

DIM_OUTPUT_PROJECTION = 32 # output projection used for approximate jacobian calculation


NUM_ITERS = 10               # number of iterations

NUM_FACTORS = 512           # number of factors to learn
FACTOR_BATCH_SIZE = 128       # factor batch size - important!

# these still work with llama 3.2 3b!
SOURCE_LAYER_IDX = 10       # source layer
TARGET_LAYER_IDX = 20       # target layer

SYSTEM_PROMPT = "You are a helpful assistant" # system prompt; set to None for no system prompt

TOKEN_IDXS = slice(-3,None) # target token positions

NUM_EVAL = 128               # number of steering vectors to evaluate



# Cell
# will use beginning/end of this dataset for train/test prompts
import pandas as pd
import random
from huggingface_hub import login

random.seed(325)

hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
if hf_token:
    login(token=hf_token)

# Load the dataset from the local file path
dataset = pd.read_csv("../tests/training_datasets/personality/essays_cAGR_y.csv")
instructions = dataset['goal'].tolist()

random.shuffle(instructions)

'''
if NUM_SAMPLES == 1:
    instructions = ["Tell me how to make a bomb."] + instructions
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True, padding_side="left",truncation_side="left")
tokenizer.pad_token = tokenizer.eos_token

if SYSTEM_PROMPT is not None:
    chat_init = [{'content':SYSTEM_PROMPT, 'role':'system'}]
else:
    chat_init = []
chats = [chat_init + [{'content': content, 'role':'user'}] for content in instructions[:NUM_SAMPLES]]
EXAMPLES = [tokenizer.apply_chat_template(chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True) for chat in chats]

test_chats = [chat_init + [{'content': content, 'role':'user'}] for content in instructions[-32:]]
TEST_EXAMPLES = [tokenizer.apply_chat_template(chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True) for chat in test_chats]

# Cell
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                             device_map="cuda", 
                                             trust_remote_code=True,
                                             _attn_implementation="eager"
                                            )


# Cell

model_inputs = tokenizer(["Tell me about yourself"], return_tensors="pt", truncation=True).to(model.device)
with torch.no_grad():
    hidden_states = model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
sliced_model = dct.SlicedModel(model, start_layer=3, end_layer=5, layers_name="model.layers")
''' unsure if this does anything, but hidden states must be loaded
with torch.no_grad():
    assert torch.allclose(sliced_model(hidden_states[3]), hidden_states[5])
'''


# Cell
sliced_model = dct.SlicedModel(model, start_layer=SOURCE_LAYER_IDX, end_layer=TARGET_LAYER_IDX, layers_name="model.layers")

# Cell
for name, param in model.named_parameters():
    print(name, param.dtype)
    break  # just to check one parameter

print(next(model.parameters()).dtype)         # should be torch.float32
print(next(sliced_model.parameters()).dtype)   # should be torch.float32
print(hidden_states[SOURCE_LAYER_IDX].dtype)


# Cell
d_model = model.config.hidden_size

X = torch.zeros(NUM_SAMPLES, MAX_SEQ_LEN, d_model, device="cpu")
Y = torch.zeros(NUM_SAMPLES, MAX_SEQ_LEN, d_model, device="cpu")

for t in tqdm(range(0, NUM_SAMPLES, FORWARD_BATCH_SIZE)):
    with torch.no_grad():
        model_inputs = tokenizer(EXAMPLES[t:t+FORWARD_BATCH_SIZE], return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_SEQ_LEN).to(model.device)
        hidden_states = model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
        h_source = hidden_states[SOURCE_LAYER_IDX] # b x t x d_model
        unsteered_target = sliced_model(h_source) # b x t x d_model

        X[t:t+FORWARD_BATCH_SIZE, :, :] = h_source
        Y[t:t+FORWARD_BATCH_SIZE, :, :] = unsteered_target



# Cell
delta_acts_single = dct.DeltaActivations(sliced_model, target_position_indices=TOKEN_IDXS) # d_model, batch_size x seq_len x d_model, batch_size x seq_len x d_model
# -> batch_size x d_model
delta_acts = vmap(delta_acts_single, in_dims=(1,None,None), out_dims=2,
                  chunk_size=FACTOR_BATCH_SIZE) # d_model x num_factors -> batch_size x d_model x num_factors

# Cell
steering_calibrator = dct.SteeringCalibrator(target_ratio=.5)

# Cell
%%time


if INPUT_SCALE is None:
    print("Model param dtype:", next(model.parameters()).dtype)
    print("X.dtype =", X.dtype, "X.device =", X.device)
    print("Y.dtype =", Y.dtype, "Y.device =", Y.device)
    print("delta_acts_single.device =", delta_acts_single.device)

    INPUT_SCALE = steering_calibrator.calibrate(delta_acts_single,X.cuda(),Y.cuda(),factor_batch_size=FACTOR_BATCH_SIZE)


# Cell
print(INPUT_SCALE)

# Cell
%%time
exp_dct= dct.ExponentialDCT(num_factors=NUM_FACTORS)
U,V = exp_dct.fit(delta_acts_single, X, Y, batch_size=BACKWARD_BATCH_SIZE, factor_batch_size=FACTOR_BATCH_SIZE,
            init="jacobian", d_proj=DIM_OUTPUT_PROJECTION, input_scale=INPUT_SCALE, max_iters=10, beta=1.0)

# Cell
from matplotlib import pyplot as plt
plt.plot(exp_dct.objective_values)

# Cell
with torch.no_grad():
    simu = (U.t() @ U)
    simu = simu[torch.triu(torch.ones_like(simu), diagonal=1).bool()]
import seaborn as sns
sns.displot(simu.cpu())


# Cell
with torch.no_grad():
    simv = (V.t() @ V)
    simv = simv[torch.triu(torch.ones_like(simv), diagonal=1).bool()]
import seaborn as sns
sns.displot(simv.cpu())


# Cell
model_inputs = tokenizer(EXAMPLES[:1], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=64, do_sample=False)
completion = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)[0]
print(completion)


# Cell
slice_to_end = dct.SlicedModel(model, start_layer=SOURCE_LAYER_IDX, end_layer=model.config.num_hidden_layers-1, 
                               layers_name="model.layers")
delta_acts_end_single = dct.DeltaActivations(slice_to_end) 

# Cell
SORRY_TOKEN = tokenizer.encode("Sorry", add_special_tokens=False)[0]
SURE_TOKEN = tokenizer.encode("Sure", add_special_tokens=False)[0]
with torch.no_grad():
    target_vec = model.lm_head.weight.data[SURE_TOKEN,:] - model.lm_head.weight.data[SORRY_TOKEN,:]

# Cell
# modified to use None rather than target_vec
scores, indices = exp_dct.rank(delta_acts_end_single, X, Y, target_vec=None, 
                               batch_size=FORWARD_BATCH_SIZE, factor_batch_size=FACTOR_BATCH_SIZE)

# Cell
import seaborn as sns
sns.displot(scores.cpu())

# Cell
model_editor = dct.ModelEditor(model, layers_name="model.layers")

# Cell
from torch import nn

# Cell
NUM_EVAL = 64
MAX_NEW_TOKENS = 16

# Cell
V_rand = torch.nn.functional.normalize(torch.randn(3072,NUM_EVAL),dim=0)
completions = []
prompt = EXAMPLES[0]
for i in tqdm(range(NUM_EVAL)):
    model_editor.restore()
    model_editor.steer(INPUT_SCALE*V_rand[:,i], SOURCE_LAYER_IDX)
    generated_ids = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    completion = tokenizer.batch_decode(generated_ids, 
                           skip_special_tokens=True)[0]
    completions.append(completion)

# Cell
for i in range(NUM_EVAL):
    print("====Random Vector %d, Positive :=========\n" % i)
    print(completions[i])


# Cell
MAX_NEW_TOKENS = 128
from torch import nn

# Cell
model_editor.restore()
completions = []
prompt = EXAMPLES[0]
for i in tqdm(range(NUM_EVAL)):
    model_editor.restore()
    model_editor.steer(INPUT_SCALE*V[:,indices[i]], SOURCE_LAYER_IDX)
    generated_ids = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    completion = tokenizer.batch_decode(generated_ids, 
                           skip_special_tokens=True)[0]
    completions.append(completion)

# Cell
for i in range(NUM_EVAL):
    print("====Steered by vector %d=========\n" % i)
    print(completions[i])


# Cell
model_editor.restore()
examples = EXAMPLES[:2] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print("=====Unsteered completion======")
    print(cont)


# Cell
VECIND = 0
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:2] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# Cell
VECIND = 68
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:2] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# Cell
VECIND = 77
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:2] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# Cell
VECIND = 91
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:2] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# Cell


