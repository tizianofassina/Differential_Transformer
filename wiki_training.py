import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import os
from pathlib import Path
import inspect
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from language_model import LanguageModel
import json

##############################################
# Step 1: Load the WikiText-2 Raw Dataset
##############################################
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_text = dataset['train']['text']
valid_text = dataset['validation']['text']
test_text = dataset['test']['text']

# The dataset is a list of strings (lines). We can concatenate them.
# Note: Each line could be a separate sentence or paragraph. 
# For simplicity, we just join them. 
# We remove empty lines to avoid excessive <eos> tokens.

train_text = [line.strip() for line in train_text if line.strip() != '']  # from [" Hello ", "  ", "World"] to ["Hello", "World"]
valid_text = [line.strip() for line in valid_text if line.strip() != '']
test_text = [line.strip() for line in test_text if line.strip() != '']

train_text_str = " ".join(train_text ) # from [" Hello ", "  ", "World"] to "Hello World"
valid_text_str = " ".join(valid_text)
test_text_str = " ".join(test_text)

##############################################
# Step 2: Tokenization. Maybe should be interesting to consider smaller tokens using an existing embedding
##############################################

train_tokens = train_text_str.split()
valid_tokens = valid_text_str.split()
test_tokens = test_text_str.split()

##############################################
# Step 3: Build Vocabulary from Training Tokens
##############################################

counter = Counter(train_tokens)

# You can limit vocabulary size if you want; here we use all tokens
# If very large, consider limiting or using subwords.

vocab = sorted(counter.keys())
token2idx = {token: i for i, token in enumerate(vocab)}
idx2token = {i: token for token, i in token2idx.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

##############################################
# Step 4: Numericalize the Datasets
##############################################

def numericalize(tokens, token2idx):
    return [token2idx[t] for t in tokens if t in token2idx]

train_ids = numericalize(train_tokens, token2idx)
valid_ids = numericalize(valid_tokens, token2idx)
test_ids = numericalize(test_tokens, token2idx)

##############################################
# Create a PyTorch Dataset and DataLoader
##############################################

# We will create a dataset that, given a list of token IDs, splits it into
# sequences of length `seq_len`, returning (input, target) pairs where target
# is the input shifted by one token.

class LMSequenceDataset(Dataset):
    def __init__(self, ids, seq_len=35):
        self.ids = ids
        self.seq_len = seq_len
        self.num_sequences = len(ids) // seq_len

    def __len__(self):
        return self.num_sequences - 1  # last one can't form a pair
    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.ids[start : start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[start + 1 : start + self.seq_len + 1], dtype=torch.long)
        return x, y

seq_len = 40
batch_size = 32

train_dataset = LMSequenceDataset(train_ids, seq_len)
valid_dataset = LMSequenceDataset(valid_ids, seq_len)
test_dataset = LMSequenceDataset(test_ids, seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


##############################################
# Model Training
##############################################

version = "10_layers_seq_len_40"
writer = SummaryWriter(f"outputs/wiki_training/normal_attention/{version}")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("device is: ", device)

mod_path = Path(inspect.getfile(LMSequenceDataset)).parent

with (open(mod_path / "models/hyper.json", "r") as f):
    f = json.load(f)

vocab_size = f["vocab_size"]
embed_dim = f["embed_dim"]
hidden_dim = f["hidden_dim"]
num_heads = f["num_heads"]
head_dim = f["head_dim"]
num_layers = f["num_layers"]
l_init = f["l_init"]
seq_len = f["seq_len"]

weight_path = mod_path / "models/wiki_training_transformer_seq_150.pth"

weight_path_diff = mod_path / "models/wiki_training_differentialtransformer_seq_150.pth"


model = LanguageModel(vocab_size, embed_dim, hidden_dim, num_heads, head_dim, num_layers, l_init, max_seq_len=seq_len, device = device).to(device)
model_diff = LanguageModel(vocab_size, embed_dim, hidden_dim, num_heads, head_dim, num_layers, l_init, diff=True, max_seq_len=seq_len, device = device).to(device)

if os.path.exists(weight_path):
    try :
        print(f"Loading weights from {weight_path}...")
        model.load_state_dict(torch.load(weight_path, map_location=torch.device(device), weights_only=True))
    except :
        print("Couldn't upload the weights, maybe you change some hyperparameters ?")
else:
    print(f"No weights file found at {weight_path} for transformer model. Starting from scratch.")

if os.path.exists(weight_path_diff):
    try :
        print(f"Loading weights from {weight_path_diff}...")
        model_diff.load_state_dict(torch.load(weight_path_diff, map_location=torch.device(device), weights_only=True))
    except :
        print("Couldn't upload the weights, maybe you change some hyperparameters ?")

else:
    print(f"No weights file found at {weight_path_diff} for differential transformer model. Starting from scratch.")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer_diff = optim.Adam(model_diff.parameters(), lr=1e-3)


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():

        for x, y in dataloader:

            x, y = x.to(device), y.to(device)
            logits = model(x)  # [batch, seq_len, vocab_size]
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
            count += 1

    return total_loss / count

def generate_from_prompt(model, prompt_str, token2idx, idx2token, generated_length=20):

    model.eval()

    prompt_tokens = prompt_str.split()
    prompt_ids = [token2idx[t] for t in prompt_tokens if t in token2idx]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(input_ids, generated_length=generated_length)
    generated_tokens = [idx2token[idx.item()] for idx in generated[0]]

    return " ".join(generated_tokens)

num_epochs = 10
global_step = 0
for epoch in tqdm(range(num_epochs)):

    model.train()
    model_diff.train()
    total_loss = 0
    total_loss_diff = 0
    count = 0

    for i, (x, y) in enumerate(tqdm(train_loader, leave=False)):

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        optimizer_diff.zero_grad()

        logits = model(x)
        logits_diff = model_diff(x)

        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss_diff = criterion(logits_diff.view(-1, vocab_size), y.view(-1))

        loss.backward()
        loss_diff.backward()

        optimizer.step()
        optimizer_diff.step()

        total_loss += loss.item()
        total_loss_diff += loss_diff.item()

        count += 1
        global_step += 1

        if (i+1) % 100 == 0:
            avg_loss = total_loss / count
            writer.add_scalar("Train/Loss Plain Model", avg_loss, global_step)
            avg_loss_diff = total_loss_diff / count
            writer.add_scalar("Train/Loss Differential Model", avg_loss_diff, global_step)

    val_loss = evaluate(model, valid_loader)
    val_loss_diff = evaluate(model_diff, valid_loader)

    writer.add_scalar("Val/Loss Plain Model", val_loss, epoch + 1)
    writer.add_scalar("Val/Loss Differential Model", val_loss_diff, epoch + 1)

test_loss = evaluate(model, test_loader)
test_loss_diff = evaluate(model_diff, test_loader)

writer.add_scalar("Test/Loss Plain Model", test_loss)
writer.add_scalar("Test/Loss Differential Model", test_loss_diff)

writer.close()

torch.save(model.state_dict(), weight_path)
torch.save(model_diff.state_dict(), weight_path_diff)


##############################################
# Test Generation
##############################################

prompt_str = "The meaning of life is"
generated_text = generate_from_prompt(model, prompt_str, token2idx, idx2token, generated_length=20)
generated_text_diff = generate_from_prompt(model_diff, prompt_str, token2idx, idx2token, generated_length=20)
print("Given Prompt:", prompt_str)
print("Generated text by Plain Model:", generated_text)
print("Generated text by Differential Model::", generated_text_diff)



