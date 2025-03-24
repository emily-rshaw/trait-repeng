#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jupyter nbconvert --to script demo_32b.ipynb')


# In[1]:


import dct
from tqdm import tqdm
import math
from torch import vmap
import torch
torch.set_default_device("cuda")
torch.manual_seed(325)


# hyper-parameters

# In[2]:


MODEL_NAME = "Qwen/Qwen1.5-32B-Chat"
TOKENIZER_NAME = MODEL_NAME

INPUT_SCALE = None           # input scale; set to None to use automatic calibration

NUM_SAMPLES = 1             # number of training samples
FORWARD_BATCH_SIZE = 1      # batch size to use when computing forward passes
BACKWARD_BATCH_SIZE = 1     # batch size to use when computing backward passes (usually smaller)
MAX_SEQ_LEN = 27            # max length for truncating training examples

CALIBRATION_SAMPLE_SIZE = 30        # sample size for random directions used to calibrate input scale
CALIBRATION_PROMPT_SAMPLE_SIZE = 1  # prompt sample size for calibrating input scale

DIM_OUTPUT_PROJECTION = 32 # output projection used for approximate jacobian calculation


NUM_ITERS = 10               # number of iterations

NUM_FACTORS = 512           # number of factors to learn
FACTOR_BATCH_SIZE = 64       # factor batch size

SOURCE_LAYER_IDX = 10       # source layer
TARGET_LAYER_IDX = 20       # target layer

SYSTEM_PROMPT = "You are a helpful assistant" # system prompt; set to None for no system prompt

TOKEN_IDXS = slice(-3,None) # target token positions

NUM_EVAL = 64               # number of steering vectors to evaluate



# In[ ]:





# train/test data. replace instructions with a data-set of your choice

# In[3]:


# will use beginning/end of this dataset for train/test prompts
import pandas as pd
import requests
import io
url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
response = requests.get(url)

dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
instructions = dataset['goal'].tolist()

import random
random.seed(325)
#random.shuffle(instructions)
if NUM_SAMPLES == 1:
    instructions = ["Tell me how to make a bomb."] + instructions


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


# # load model

# In[4]:


get_ipython().run_cell_magic('time', '', 'from transformers import AutoModelForCausalLM, AutoTokenizer\nmodel = AutoModelForCausalLM.from_pretrained(MODEL_NAME,\n                                             device_map="auto", \n                                             trust_remote_code=True, \n                                             _attn_implementation="eager"\n                                            )\n')


# In[ ]:





# # slice model

# In[5]:


if SOURCE_LAYER_IDX is None:
    SOURCE_LAYER_IDX = model.config.num_hidden_layers // 2 - 5
if TARGET_LAYER_IDX is None:
    TARGET_LAYER_IDX = model.config.num_hidden_layers // 2 + 5


# check we're correctly calculating hidden states

# In[6]:


model_inputs = tokenizer(["tell me how to make a bomb"], return_tensors="pt", truncation=True).to(model.device)
with torch.no_grad():
    hidden_states = model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
sliced_model =dct.SlicedModel(model, start_layer=3, end_layer=5, layers_name="model.layers")
with torch.no_grad():
    assert torch.allclose(sliced_model(hidden_states[3]), hidden_states[5])


# slice we'll actually use

# In[7]:


sliced_model = dct.SlicedModel(model, start_layer=SOURCE_LAYER_IDX, end_layer=TARGET_LAYER_IDX, layers_name="model.layers")


# In[ ]:





# # construct dataset of unsteered activations

# In[8]:


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



# # class computing $\Delta_{\mathcal A}$

# In[9]:


delta_acts_single = dct.DeltaActivations(sliced_model, target_position_indices=TOKEN_IDXS) # d_model, batch_size x seq_len x d_model, batch_size x seq_len x d_model
# -> batch_size x d_model
delta_acts = vmap(delta_acts_single, in_dims=(1,None,None), out_dims=2,
                  chunk_size=FACTOR_BATCH_SIZE) # d_model x num_factors -> batch_size x d_model x num_factors


# # calibrate $R$

# In[10]:


steering_calibrator = dct.SteeringCalibrator(target_ratio=.5)


# In[11]:


get_ipython().run_cell_magic('time', '', 'if INPUT_SCALE is None:\n    INPUT_SCALE = steering_calibrator.calibrate(delta_acts_single,X.cuda(),Y.cuda(),factor_batch_size=FACTOR_BATCH_SIZE)\n')


# In[12]:


print(INPUT_SCALE)


# # Fit Exponential DCT

# In[13]:


get_ipython().run_cell_magic('time', '', 'exp_dct= dct.ExponentialDCT(num_factors=NUM_FACTORS)\nU,V = exp_dct.fit(delta_acts_single, X, Y, batch_size=BACKWARD_BATCH_SIZE, factor_batch_size=FACTOR_BATCH_SIZE,\n            init="jacobian", d_proj=DIM_OUTPUT_PROJECTION, input_scale=INPUT_SCALE, max_iters=10, beta=1.0)\n')


# In[14]:


from matplotlib import pyplot as plt
plt.plot(exp_dct.objective_values)


# # Visualize factor similarities

# In[15]:


with torch.no_grad():
    simu = (U.t() @ U)
    simu = simu[torch.triu(torch.ones_like(simu), diagonal=1).bool()]
import seaborn as sns
sns.displot(simu.cpu())


# In[16]:


with torch.no_grad():
    simv = (V.t() @ V)
    simv = simv[torch.triu(torch.ones_like(simv), diagonal=1).bool()]
import seaborn as sns
sns.displot(simv.cpu())


# # Evaluate

# ## Unsteered

# In[17]:


model_inputs = tokenizer(EXAMPLES[:1], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=64, do_sample=False)
completion = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)[0]
print(completion)


# ## Rank factors

# In[18]:


slice_to_end = dct.SlicedModel(model, start_layer=SOURCE_LAYER_IDX, end_layer=model.config.num_hidden_layers-1, 
                               layers_name="model.layers")
delta_acts_end_single = dct.DeltaActivations(slice_to_end) 


# In[19]:


SORRY_TOKEN = tokenizer.encode("Sorry", add_special_tokens=False)[0]
SURE_TOKEN = tokenizer.encode("Sure", add_special_tokens=False)[0]
with torch.no_grad():
    target_vec = model.lm_head.weight.data[SURE_TOKEN,:] - model.lm_head.weight.data[SORRY_TOKEN,:]


# In[20]:


scores, indices = exp_dct.rank(delta_acts_single, X, Y, target_vec=target_vec, 
                               batch_size=FORWARD_BATCH_SIZE, factor_batch_size=FACTOR_BATCH_SIZE)


# In[21]:


import seaborn as sns
sns.ecdfplot(scores.cpu(), complementary=True, log_scale=(True,True))


# # save

# In[33]:


#torch.save(U, "14B_n1_U.pt")
#torch.save(V, "14B_n1_V.pt")


# # Evaluate Model Edits

# In[22]:


model_editor = dct.ModelEditor(model, layers_name="model.layers")


# ## Steer

# In[23]:


NUM_EVAL = 32
MAX_NEW_TOKENS = 128
from torch import nn


# In[24]:


model_editor.restore()
completions = []
prompt = EXAMPLES[0]
for i in tqdm(range(NUM_EVAL)):
    model_editor.steer(INPUT_SCALE*V[:,indices[i]], SOURCE_LAYER_IDX)
    generated_ids = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    completion = tokenizer.batch_decode(generated_ids, 
                           skip_special_tokens=True)[0]
    completions.append(completion)


# In[25]:


for i in range(NUM_EVAL):
    print("====Steer with Vector %d, Positive :=========\n" % i)
    print(completions[i])


# ## Detailed eval: unsteered

# In[26]:


model_editor.restore()
examples = EXAMPLES[:1] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print("=====Unsteered completion======")
    print(cont)


# ## Detailed eval: steering specific vectors

# In[28]:


VECIND = 1
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:1] + TEST_EXAMPLES[-4:]
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# In[30]:


VECIND = 10
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:1] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# In[32]:


VECIND = 26
model_editor.restore()
model_editor.steer(INPUT_SCALE*V[:,indices[VECIND]], SOURCE_LAYER_IDX)
examples = EXAMPLES[:1] + TEST_EXAMPLES
model_inputs = tokenizer(examples, return_tensors="pt", padding=True).to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
conts = tokenizer.batch_decode(generated_ids, 
                       skip_special_tokens=True)
for cont in conts:
    print(f"======Steered by vector {VECIND}=====")
    print(cont)


# In[ ]:




