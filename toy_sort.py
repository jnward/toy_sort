#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model
from tqdm import tqdm

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("cpu")
    print("Using MPS device")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MPS not available, using device: {device}")

class SortingDataset(Dataset):
    def __init__(self, size=1000, seq_length=10):
        self.size = size
        self.seq_length = seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sequence = torch.randint(0, 1000, (self.seq_length,), device=device)
        sorted_sequence = torch.sort(sequence)[0]
        
        # Construct target: [BOS, sequence, BOS, sorted_sequence, EOS]
        target_ids = torch.full((22,), 1000, device=device)  # Initialize with BOS/EOS token
        target_ids[1:11] = sequence
        target_ids[11] = 1000  # BOS token to separate input and output
        target_ids[12:22] = sorted_sequence
        
        return target_ids


# In[2]:

# Create dataset and dataloader
dataset = SortingDataset()
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


# In[3]:


from tqdm import tqdm

def train(model, dataloader, optimizer, num_epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch, target_ids in enumerate(progress_bar):
            optimizer.zero_grad()

            input_ids = target_ids[..., :-1]
            output_ids = target_ids[..., 11:]
            
            outputs = model(input_ids)[..., -11:, :]
            loss = criterion(outputs.reshape(-1, model.cfg.d_vocab), output_ids.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if batch%100 == 1:
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    return epoch_losses


# In[4]:
def sort_sequence(model, sequence):
    model.eval()
    with torch.no_grad():
        # Prepare input: [BOS, sequence, BOS, zeros]
        input_ids = torch.full((1, 22), 1000, device=device)  # Initialize with BOS/EOS token
        input_ids[0, 1:11] = sequence
        print(input_ids)
        
        # Generate sorted sequence
        for i in range(11, 22):
            outputs = model(input_ids[:, :i])
            next_token = outputs[0, -1, :].argmax()
            input_ids[0, i] = next_token
        
        sorted_sequence = input_ids[0, -10:].cpu()
    
    return sorted_sequence


# In[5]:
from transformer_lens import HookedTransformer, HookedTransformerConfig

hooked_config = HookedTransformerConfig(
    n_layers=1,
    d_model=64,
    n_ctx=22,
    d_head=64,
    attn_only=True,
    d_vocab=1001,
    device=device,
    act_fn="gelu_new",
    normalization_type=None,
)

hooked_model = HookedTransformer(hooked_config)

hooked_model.unembed.W_U.data = hooked_model.embed.W_E.T


# In[6]:


path = "models/tiny_sort_1_0.056.pt"
checkpoint = torch.load(path, weights_only=True)
hooked_model.load_state_dict(checkpoint)


# In[6]:


# Create dataset and dataloader
dataset = SortingDataset(size=100000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# # Initialize model and optimizer
# model = SortingTransformer(config).to(device)
optimizer = optim.Adam(hooked_model.parameters(), lr=1e-3)

train(hooked_model, dataloader, optimizer, 5)
for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-4
train(hooked_model, dataloader, optimizer, 15)

seq = torch.randint(0, 1000, (10,))
sorted_seq = sort_sequence(hooked_model, seq)
expected_sequence = seq.sort().values
print(f"Original sequence: {seq}")
print(f"Sorted sequence:   {sorted_seq}")
print(f"Expected sequence: {expected_sequence}")

# 1.76 with 2L 2H 32 d_model


# In[53]:


for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-6
arr = train(hooked_model, dataloader, optimizer, 100)


# In[81]:


# path = "models/tiny_sort_1_0.056.pt"

# torch.save(hooked_model.state_dict(), path)


# In[26]:


seq = torch.randint(0, 1000, (10,))
sorted_seq = sort_sequence(hooked_model, seq)
expected_sequence = seq.sort().values
print(f"Original sequence: {seq}")
print(f"Sorted sequence:   {sorted_seq}")
print(f"Expected sequence: {expected_sequence}")


# In[27]:


import circuitsvis as cv

def get_random_seq():
    sequence = torch.randint(0, 1000, (10,))
    sorted_sequence = torch.sort(sequence)[0]
    
    target_ids = torch.full((22,), 1000)
    target_ids[1:11] = sequence
    target_ids[12:22] = sorted_sequence

    return target_ids

seq = get_random_seq()
seq


# In[29]:


def tok_to_str(token_ids):
    return list(map(str, token_ids.cpu().tolist()))

# test_seq = torch.arange(50, 1000, 100)
test_seq = torch.tensor([0, 100, 150, 200, 250, 750, 800, 850, 900, 950])
# test_seq = torch.tensor([150, 100, 850, 750, 200, 250, 900, 0, 800, 950])
# test_seq = torch.tensor([0, 100, 150, 200, 230, 245, 250, 250, 250, 950])
# test_seq = torch.tensor([0, 50, 800, 820, 840, 860, 880, 900, 920, 950])

seq[1:11] = test_seq
seq[12:] = torch.sort(test_seq)[0]
# seq[1:11] = seq[12:]
# seq[1:11] = torch.flip(seq[12:], (0,))

logits, cache = hooked_model.run_with_cache(seq)

strs = tok_to_str(seq)
attention_pattern = cache["pattern", 0, "attn"]
print(strs)
print(attention_pattern.shape)

cv.attention.attention_patterns(tokens=strs, attention=attention_pattern[0])


# In[32]:


def get_resid_component_logits(component):
    return (component @ hooked_model.unembed.W_U + hooked_model.unembed.b_U)[0]

def print_str_vals(strs, vals):
    for s, val in zip(strs, vals):
        print(f"{s}: {val:.4f}")

attn_logits = get_resid_component_logits(cache["attn_out", 0])
print_str_vals(strs, attn_logits.argmax(1))


# tokens attend to:
# - strongly the next greatest number (unsorted)
# - their unsorted counterpart
# - smaller numbers (unsorted)
# 
# the 250 token attends most strongly to it's unsorted counterpart, 800, then 750. How does it predict 750?
# - maybe it attends to itself (current token) weakly?
# 

# In[33]:


# let's look directly at the attention pattern for the sorted 250 token

print_str_vals(strs, attention_pattern[0, 0, 16, :])


# it does attend to itself slightly. Is this used in the OV circuit?
# 
# can verify by zeroing out the attention pattern for this token and running forward pass. How does this change logits?

# In[34]:


def get_attn_outputs_from_pattern(pattern, attn):
    # V = cache['embed'] @ attn.W_V + attn.b_V # this doesn't work for some reason? ->>> I forgot pos embeddings
    V = cache['v', 0]
    weighted_V = V * pattern[None, :, None, None]
    z = weighted_V.sum(dim=1)
    attn_out = z @ attn.W_O + attn.b_O
    return attn_out

attn = hooked_model.blocks[0].attn
my_attn_out = get_attn_outputs_from_pattern(attention_pattern[0, 0, 16, :], attn)
cache_attn_out = cache['attn_out', 0][0, 16, :]

torch.allclose(my_attn_out, cache_attn_out)

# my function works! Now I can compute attn_out logits with a custom pattern.


# In[35]:


get_resid_component_logits(my_attn_out).argmax()


# In[36]:


attention_pattern[0, 0, 16, :]


# In[37]:


import plotly.express as px
import pandas as pd


# In[38]:


labels = [f"{s}_{i}" for i, s in enumerate(strs)]

def graph_attn_out_diff(orig_pat, modified_pat):
    df = pd.DataFrame({
        'labels': labels + labels,
        'attn_pattern': orig_attn_pat.tolist() + modified_pat.tolist(),
        'group': ["orig"] * len(labels) + ["mod"] * len(labels),
    })
    
    px.bar(df, x='labels', y='attn_pattern', color='group', barmode='group', log_y=False).show()
    
    orig_attn_out = get_attn_outputs_from_pattern(orig_attn_pat, attn)
    mod_attn_out = get_attn_outputs_from_pattern(modified_pat, attn)
    orig_logits = get_resid_component_logits(orig_attn_out)
    mod_logits = get_resid_component_logits(mod_attn_out)
    
    px.line(orig_logits[0].tolist()).show()
    px.line(mod_logits[0].tolist()).show()


# In[39]:


orig_attn_pat = attention_pattern[0, 0, 16, :]
my_attn_pat = attention_pattern[0, 0, 16, :].clone()
my_attn_pat[12:17] = 0

graph_attn_out_diff(orig_attn_pat, my_attn_pat)


# Looks like ablating the sorted values doesn't do anything. Somehow, the correct token is still predicted.
# 
# - there must be some way the model is deriving the current position without looking at the current token. maybe this is the BOS token?

# In[41]:


orig_attn_pat = attention_pattern[0, 0, 16, :]
my_attn_pat = attention_pattern[0, 0, 16, :].clone()
my_attn_pat[12:17] = 0
my_attn_pat[0] = 0
my_attn_pat[11] = 0
print(my_attn_pat)

graph_attn_out_diff(orig_attn_pat, my_attn_pat)


# In[42]:


orig_attn_pat = attention_pattern[0, 0, 16, :]
my_attn_pat = attention_pattern[0, 0, 16, :].clone()
my_attn_pat[0] = 0
my_attn_pat[11] = 0
my_attn_pat[:6] = 0
my_attn_pat[:] = 0
my_attn_pat[6] = 1
print(my_attn_pat)

graph_attn_out_diff(orig_attn_pat, my_attn_pat)


# Nope... let's try playing with the unsorted tokens

# In[43]:


orig_attn_pat = attention_pattern[0, 0, 16, :]
my_attn_pat = attention_pattern[0, 0, 16, :].clone()
my_attn_pat[0] = 0
my_attn_pat[11] = 0
my_attn_pat[:6] = 0
my_attn_pat[:] = 0
my_attn_pat[7] = 1
print(my_attn_pat)

graph_attn_out_diff(orig_attn_pat, my_attn_pat)


# In[44]:


orig_attn_pat = attention_pattern[0, 0, 16, :]
my_attn_pat = attention_pattern[0, 0, 16, :].clone()
my_attn_pat[0] = 0
my_attn_pat[11] = 0
my_attn_pat[:6] = 0
my_attn_pat[:] = 0
my_attn_pat[5] = 1
print(my_attn_pat)

graph_attn_out_diff(orig_attn_pat, my_attn_pat)


# In[45]:


orig_attn_pat = attention_pattern[0, 0, 16, :]
my_attn_pat = attention_pattern[0, 0, 16, :].clone()
my_attn_pat[0] = 0
my_attn_pat[11] = 0
my_attn_pat[:6] = 0
my_attn_pat[:] = 0
my_attn_pat[5:8] = 1/3
print(my_attn_pat)

graph_attn_out_diff(orig_attn_pat, my_attn_pat)


# Positional embeddings shouldn't matter a lot for these unsorted tokens, since their unsorted order is random.
# 
# What I'm noticing:
# - All tokens boost themselves
# - High tokens also boost slightly lower tokens and diminish _much lower_ tokens
# - Low tokens boost slightly lower tokens and diminish slightly higher tokens
# 
# So what's going on w/ 750?
# - 250 boosts itself
# - 750 boosts itself and diminishes 250
# - 800 boosts itself and 750, and diminishes 250
# - overall, 250, 750, and 800 predict 750 for the next token
# 
# I think attention scores generally favor higher values, especially values near the current token (hence the attending to unsorted self). 750/800 are so high because of softmax.
# 
# ^ This all feels correct, but are there more rigorous ways to verify?

# In[46]:


sort_sequence(hooked_model, seq[12:])


# In[47]:


QK = hooked_model.embed.W_E @ attn.QK[0] @ hooked_model.unembed.W_U


# In[48]:


fig = px.imshow(
    QK.AB.detach().numpy(),
    labels=dict(x="Key", y="Query", color="Value"),
    title="QK Matrix")

fig.update_layout(
    autosize=False,
    width=400,
    height=400,
)

fig.show()


# In[49]:


OV = hooked_model.embed.W_E @ attn.OV[0] @ hooked_model.unembed.W_U


# In[50]:


fig = px.imshow(
    OV.AB.detach().numpy(),
    labels=dict(x="Logit", y="Vocab", color="Value"),
    title="OV Matrix")

fig.update_layout(
    autosize=False,
    width=400,
    height=400,
)

fig.show()


# ## embedding SVD

# In[60]:


tokens = torch.arange(0, 1001)

U, S, V = hooked_model.embed.W_E[tokens].svd()


# In[168]:


px.line((U * S[None, :])[:, torch.arange(0, 64)].detach().numpy())


# In[82]:


# first two look like sin and cos, all look very periodic. looks like model has learned to do f. transform?
# verify by looking at f. decomp bar chart for singular vectors


# In[104]:


pos_tokens = torch.arange(0, 22)

U_pos, S_pos, V_pos = hooked_model.pos_embed.W_pos[pos_tokens].svd()


# In[105]:


px.line((U_pos * S_pos[None, :])[:, torch.arange(0, 22)].detach().numpy())


# ## Fourier Decomposition of Embedding Singular Vectors

# In[190]:


import math

def generate_fourier_components(n):
    components = []
    labels = []
    
    constant = torch.ones(n)
    components.append(constant)
    labels.append("constant")
    
    for k in range(1, (n-1)//2 + 1):
        t = torch.arange(n) / n

        cos_vec = torch.cos(2 * math.pi * k * t)
        components.append(cos_vec)
        labels.append(f"cos_{k}")
        
        sin_vec = torch.sin(2 * math.pi * k * t)
        components.append(sin_vec)
        labels.append(f"sin_{k}")
    
    if n % 2 == 0:
        sign_vec = torch.tensor([(1 if j%2==0 else -1) for j in range(n)])
        components.append(sign_vec)
        labels.append("sign")
    
    components_tensor = torch.stack(components)
    components_tensor = components_tensor / components_tensor.norm(dim=-1, keepdim=True)
    
    return components_tensor, labels

def fourier_basis(n):
    terms = []
    labels = []
    terms.append(torch.ones(n))
    labels.append("const")
    for i in range(1, (n-1)//2 + 1):
        terms.append(torch.cos(torch.arange(n) * 2 * torch.pi / n * i))
        terms.append(torch.sin(torch.arange(n) * 2 * torch.pi / n * i))
        labels.append(f"cos_{i}")
        labels.append(f"sin_{i}")
    if n%2 == 0:
        terms.append(torch.tensor([(1 if j%2==0 else -1) for j in range(n)]))
        labels.append("sign")
    terms = torch.stack(terms)
    terms = terms / terms.norm(dim=-1, keepdim=True)
    return terms, labels


# In[191]:


(U * S[None, :]).shape


# In[237]:


# fourier_components, labels = generate_fourier_components(1001)
fourier_components, labels = fourier_basis(1001)
print(fourier_components.shape, len(labels))

f_basis = fourier_components @ (U * S)
f_basis.shape


# In[223]:


px.line(fourier_components[4, :].detach().numpy())


# In[231]:


px.line(U[:, 4].detach().numpy())


# In[230]:


px.line(torch.stack([U[:, 4], -fourier_components[4, :]]).T.detach().numpy())


# In[222]:


px.bar(y=f_basis[:100, 4].detach().numpy(), x=labels[:100])


# In[243]:


px.line(f_basis[:, :40].detach().numpy())


# In[ ]:





# In[94]:


attn.QK[0].AB


# In[128]:


attn.W_V[0] @ attn.W_O[0]


# In[124]:


attn.OV[0].AB


# In[ ]:




