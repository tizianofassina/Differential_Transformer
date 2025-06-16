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
from language_model import LanguageModel, LanguageModel_PretrainedEmb
import json
from transformers import RobertaTokenizer, RobertaForCausalLM

mod_path = Path(inspect.getfile(inspect.currentframe())).parent

##############################################
# Step 1: Load the WikiText-2 Raw Dataset
##############################################

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
print("Dataset loaded")

train_text = dataset['train']['text']
train_text = [t for t in train_text if t.strip() != ""]
print("Lines of the dataset:", len(train_text))
print(train_text[567])

valid_text = dataset['validation']['text']
valid_text = [t for t in valid_text if t.strip() != ""]

test_text = dataset['test']['text']
test_text = [t for t in test_text if t.strip() != ""]

train_text_str = " ".join(train_text) # from [" Hello ", "  ", "World"] to "Hello World"
valid_text_str = " ".join(valid_text)
test_text_str = " ".join(test_text)


##############################################
# Step 2: Tokenization with RoBERTa Tokenizer
##############################################

class TextDataset(Dataset):
    def __init__(self, list_of_texts, tokenizer, max_length=40):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for text in list_of_texts:
            enc = self.tokenizer(
                text,
                add_special_tokens=True,  # RoBERTa typically uses <s> and </s>
                truncation=True,
                max_length=self.max_length
            )
            input_ids = enc["input_ids"]
            self.examples.append(input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return a 1D list of token IDs
        return torch.tensor(self.examples[idx], dtype=torch.long)

def collate_fn(batch):
    # batch is a list of 1D tensors of variable lengths
    # we want to pad them to the same length
    input_ids = torch.nn.utils.rnn.pad_sequence(
        batch, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
vocab_size = tokenizer.vocab_size

with (open(mod_path / "models/hyper_roberta.json", "r") as f):
    f = json.load(f)

seq_len = f["seq_len"]
f["vocab_size"] = vocab_size
print("Vocab size:", vocab_size)

train_dataset = TextDataset(train_text, tokenizer, max_length=seq_len)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)

valid_dataset = TextDataset(valid_text, tokenizer, max_length=seq_len)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)

test_dataset = TextDataset(test_text, tokenizer, max_length=seq_len)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)

print("DataLoaders created.")


##############################################
# Model Training
##############################################

version = "10_layers_seq_len_40_hidden_128_good_params"
writer = SummaryWriter(f"outputs/wiki_big/{version}")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("device is: ", device)

vocab_size = f["vocab_size"]
embed_dim = f["embed_dim"]
hidden_dim = f["hidden_dim"]
num_heads = f["num_heads"]
head_dim = f["head_dim"]
num_layers = f["num_layers"]
l_init = f["l_init"]
seq_len = f["seq_len"]

roberta = RobertaForCausalLM.from_pretrained("roberta-base")

# Get the pretrained embedding
input_emb = roberta.get_input_embeddings().to(device)

# Freeze the embedding weights
for param in input_emb.parameters():
    param.requires_grad = False

output_emb = roberta.get_output_embeddings().to(device)
for param in output_emb.parameters():
    param.requires_grad = False

weight_path = mod_path / "models/wiki_training_transformer_roberta.pth"
weight_path_diff = mod_path / "models/wiki_training_differentialtransformer_roberta.pth"


model = LanguageModel_PretrainedEmb(input_emb, output_emb, vocab_size, embed_dim, hidden_dim, num_heads, head_dim, num_layers, l_init, max_seq_len=seq_len, device = device).to(device)
model_diff = LanguageModel_PretrainedEmb(input_emb, output_emb, vocab_size, embed_dim, hidden_dim, num_heads, head_dim, num_layers, l_init, diff=True, max_seq_len=seq_len, device = 'cpu')

model_diff = model_diff.to(device)


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
        model_diff.load_state_dict(torch.load(weight_path_diff, map_location=torch.device(device), weights_only=True), strict=False)
    except :
        print("Couldn't upload the weights, maybe you change some hyperparameters?")

else:
    print(f"No weights file found at {weight_path_diff} for differential transformer model. Starting from scratch.")


criterion = nn.functional.cross_entropy
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
optimizer_diff = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_diff.parameters()), lr=5e-4)


def evaluate(model, dataloader):

    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            logits = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
            shift_logits = logits.contiguous()  # [B, T-1, vocab_size]
            shift_labels = input_ids[:, 1:].contiguous()   # [B, T-1]
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = criterion(shift_logits, shift_labels, ignore_index=tokenizer.pad_token_id)
            total_loss += loss.item()
            count += 1
    
    return total_loss / count


def generate_from_prompt(model, prompt_str, tokenizer, generated_length=20):
    model.eval()

    # Tokenize the input prompt using RoBERTa tokenizer
    input_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors='pt').to(device)

    # Generate tokens from the model
    generated_ids = model.generate(input_ids, generated_length=generated_length)

    # Decode the generated token IDs back into text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

num_epochs = 10
global_step = 0

for epoch in tqdm(range(num_epochs)):

    model.train()
    model_diff.train()
    total_loss = 0
    total_loss_diff = 0
    count = 0

    for step, (input_ids, attention_mask) in enumerate(tqdm(train_loader, leave=False)):

        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        # logits = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
        logits_diff  =  model_diff(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
        # shift by 1 for next-token prediction
        
        
        # we ignore the last token's prediction or the first token's label
        # shift_logits = logits.contiguous()  # [B, T-1, vocab_size]
        shift_labels = input_ids[:, 1:].contiguous()   # [B, T-1]
        shift_logits_diff = logits_diff.contiguous()
        
        # flatten for cross-entropy
        # shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_logits_diff = shift_logits_diff.view(-1, shift_logits_diff.size(-1))

        shift_labels = shift_labels.view(-1)

        # ignore pad tokens
        # loss = criterion(shift_logits, shift_labels, ignore_index=tokenizer.pad_token_id)
        loss_diff = criterion(shift_logits_diff, shift_labels, ignore_index=tokenizer.pad_token_id)
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        optimizer_diff.zero_grad()
        loss_diff.backward()
        optimizer_diff.step()

        # total_loss += loss.item()
        total_loss_diff += loss_diff.item()

        count += 1
        global_step += 1

        if (step+1) % 1000 == 0:
            avg_loss = total_loss / count
            # writer.add_scalar("Train/Loss Plain Model", avg_loss, global_step)
            avg_loss_diff = total_loss_diff / count
            writer.add_scalar("Train/Loss Differential Model", avg_loss_diff, global_step)
            count = 0
            total_loss = 0
            total_loss_diff = 0

    val_loss = evaluate(model, valid_loader)
    val_loss_diff = evaluate(model_diff, valid_loader)
    writer.add_scalar("Val/Loss Plain Model", val_loss, epoch + 1)
    writer.add_scalar("Val/Loss Differential Model", val_loss_diff, epoch + 1)

    prompt_str = "The human"
    generated_text = generate_from_prompt(model, prompt_str, tokenizer, generated_length=60)
    print(" Given Prompt:", prompt_str)
    print("Generated text by Plain Model:", generated_text)
    generated_text_diff = generate_from_prompt(model_diff, prompt_str, tokenizer, generated_length=60)
    print("Generated text by Differential Model:", generated_text_diff)

test_loss = evaluate(model, test_loader)
test_loss_diff = evaluate(model_diff, test_loader)

print(f"Test Loss: {test_loss:.4f}, Test PPL: {torch.exp(torch.tensor(test_loss)):.2f}")
writer.add_scalar("Test/Loss Plain Model", test_loss)
writer.add_scalar("Test/Loss Differential Model", test_loss_diff)

writer.close()

torch.save(model.state_dict(), weight_path)
torch.save(model_diff.state_dict(), weight_path_diff)


##############################################
# Test Generation
##############################################

prompt_str = "Mathematics are"

generated_text = generate_from_prompt(model, prompt_str, tokenizer, generated_length=100)
generated_text_diff = generate_from_prompt(model_diff, prompt_str, tokenizer, generated_length=100)

print("Given Prompt:", prompt_str)
print("Generated text by Plain Model:", generated_text)
print("Generated text by Differential Model::", generated_text_diff)



