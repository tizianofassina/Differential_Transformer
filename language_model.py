import torch
import torch.nn as nn
import json
import os
from pathlib import Path
import inspect

from model import Transformer, DifferentialTransformer



class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, head_number, head_dim, num_layers, l_init, diff = False,max_seq_len=128, device='cpu'):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding and positional embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.proj = nn.Linear(embed_dim, hidden_dim)

        if diff:
            self.transformer = DifferentialTransformer(hidden_dim, head_number, head_dim, num_layers, l_init, device = device)
        else:
            self.transformer = Transformer(hidden_dim, head_number, head_dim, num_layers, l_init, device = device)

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):

        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device) # [seq_len]

        tok_emb = self.token_embedding(input_ids) # [batch_size, seq_len, embed_dim]
        x = self.proj(tok_emb)
        pos_emb = self.position_embedding(positions.unsqueeze(0)) # [1, seq_len, hidden_dim]
        x = x + pos_emb

        x = self.transformer(x, attention_mask=attention_mask)  # [batch_size, seq_len, embed_dim]
        output = self.lm_head(x) # [batch_size, seq_len, vocab_size]
        return output

    @torch.no_grad()
    def generate(self, prompt, generated_length=20):

        # prompt: [batch_size, seq_len]
        for _ in range(generated_length):

            if prompt.size(1) > self.max_seq_len:
                break

            logprobs = self.forward(prompt) # [batch_size, seq_len, vocab_size]
            next_token_logprobs = logprobs[:, -1, :]  # Last time step
            next_token = torch.argmax(next_token_logprobs, dim=-1, keepdim=True)
            prompt = torch.concat([prompt, next_token], dim=1)

        return prompt

class LanguageModel_PretrainedEmb(nn.Module):
    def __init__(self, input_emb, output_emb, vocab_size, embed_dim, hidden_dim, head_number, head_dim, num_layers, l_init, diff = False,max_seq_len=128, device='cpu'):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding and positional embedding
        self.token_embedding = input_emb


        for param in self.token_embedding.parameters():
            param.requires_grad = False

        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.proj = nn.Linear(embed_dim, hidden_dim)

        if diff:
            self.transformer = DifferentialTransformer(hidden_dim, head_number, head_dim, num_layers, l_init, device = device)
        else:
            self.transformer = Transformer(hidden_dim, head_number, head_dim, num_layers, l_init, device = device)
        self.proj_2 = nn.Linear(hidden_dim, embed_dim)

        self.lm_head = output_emb
        for param in self.lm_head.parameters():
            param.requires_grad = False


    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device) # [seq_len]

        tok_emb = self.token_embedding(input_ids) # [batch_size, seq_len, embed_dim]
        x = self.proj(tok_emb)
        pos_emb = self.position_embedding(positions.unsqueeze(0)) # [1, seq_len, hidden_dim]
        x = x + pos_emb

        x = self.proj_2(self.transformer(x, attention_mask=attention_mask))  # [batch_size, seq_len, embed_dim]
        output = self.lm_head(x) # [batch_size, seq_len, vocab_size]

        # Maybe we can include softmax after output, I am not sure if it is relevant.
        # The softmax is usually done over the last element of the sequence only however.

        return output

    # For this function we should try to generate until we get a EOS value.
    # This is a first version.

    @torch.no_grad()
    def generate(self, prompt, generated_length=20):
        # prompt: [batch_size, seq_len]

        device = prompt.device
        for _ in range(generated_length):

            if prompt.size(1) > self.max_seq_len:
                break

            logprobs = self.forward(prompt) # [batch_size, seq_len, vocab_size]
            next_token_logprobs = logprobs[:, -1, :]  # Last time step
            next_token = torch.argmax(next_token_logprobs, dim=-1, keepdim=True) # We could include softmax here too
            prompt = torch.concat([prompt, next_token], dim=1)

        return prompt

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"




    mod_path = Path(inspect.getfile(inspect.currentframe())).parent

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

    model = LanguageModel(vocab_size, embed_dim, hidden_dim, num_heads, head_dim, num_layers, l_init,
                              max_seq_len=seq_len, device=device).to(device)
    model_diff = LanguageModel(vocab_size, embed_dim, hidden_dim, num_heads, head_dim, num_layers, l_init, diff=True,
                                   max_seq_len=seq_len, device=device).to(device)

    if os.path.exists(weight_path):
        try:
            print(f"Loading weights from {weight_path}...")
            model.load_state_dict(torch.load(weight_path, map_location=torch.device(device), weights_only=True))
        except:
            print("Couldn't upload the weights, maybe you change some hyperparameters ?")
    else:
        print(f"No weights file found at {weight_path} for transformer model. Starting from scratch.")

    if os.path.exists(weight_path_diff):
        try:
            print(f"Loading weights from {weight_path_diff}...")
            model_diff.load_state_dict(
                torch.load(weight_path_diff, map_location=torch.device(device), weights_only=True))
        except:
            print("Couldn't upload the weights, maybe you change some hyperparameters ?")

    else:
        print(f"No weights file found at {weight_path_diff} for differential transformer model. Starting from scratch.")

        # print("testing:", model.transformer.diff_attentions[0].q_linear.device)


    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    logprobs = model(input_ids)
    logprobs_diff = model_diff(input_ids)

    print("Logits shape:", logprobs.shape)  # [2, 10, 50]
    print("Logits shape diff:", logprobs_diff.shape)  # [2, 10, 50]

    next_token_logprobs = logprobs[:, -1, :]
    next_token_logprobs_diff = logprobs_diff[:, -1, :]
    next_token = torch.argmax(next_token_logprobs, dim=-1)
    next_token_diff = torch.argmax(next_token_logprobs_diff, dim=-1)

    print("Next token predicted :", next_token) # length 2, 50 possible values.
    print("Next token predicted diff :", next_token_diff) # length 2, 50 possible values.


    # Test generation
    generated = model.generate(input_ids, generated_length=5)
    generated_diff = model_diff.generate(input_ids, generated_length=5)

    print("Generated sequence:", generated)
    print("Generated sequence diff :", generated_diff )

