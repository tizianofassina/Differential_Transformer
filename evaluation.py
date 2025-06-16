import os
import torch
import torch.nn.functional as F
from pathlib import Path
import inspect
from tqdm import tqdm
import json
from transformers import RobertaTokenizer, RobertaForCausalLM
from lm_eval import evaluator, tasks, models
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

from language_model import LanguageModel_PretrainedEmb

mod_path = Path(inspect.getfile(inspect.currentframe())).parent

class TestLM(LM):
    """
    A wrapper around LanguageModel_PretrainedEmb that implements 
    the LM Evaluation Harness API.
    """

    def __init__(
        self,
        pretrained_lm,     # An instance of LanguageModel_PretrainedEmb
        tokenizer,
        device="cuda",
        max_seq_len: int = 40,
    ):
        super().__init__()
        self.model = pretrained_lm
        self.model.eval()           # put in eval mode
        self.device = device
        self.model.to(device)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        request.args = (input_text, target_text)
        We sum the log-probs of each target token given the input text, 
        chunking so we never exceed 40 tokens in a forward pass.
        Returns (logprob, is_greedy).
        """
        outputs = []
        print("likelihood")
        for req in tqdm(requests):
            input_text, target_text = req.args

            # Tokenize
            input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
            all_ids = input_ids + target_ids

            # We'll sum the log-likelihood for each token in the target portion.
            total_logprob = 0.0
            is_greedy_overall = True

            # We process each token in the target portion:
            #   for j in range(prefix_len, prefix_len + len(target_ids))
            # We feed up to 39 tokens of context + 1 new token = 40 max
            prefix_len = len(input_ids)

            for idx_in_target in range(len(target_ids)):
                # The position in the combined sequence
                pos = prefix_len + idx_in_target

                # We'll create a window that includes as many of the preceding tokens
                # as possible (up to 39) plus the current token as the 40th.
                window_start = max(0, pos - self.max_seq_len + 1)
                window_end = pos + 1  # slice is exclusive on the end

                window = all_ids[window_start:window_end]   # up to 40 tokens
                # The token we want to measure is the last one in `window`
                # We'll feed everything except the last token into the model,
                # then compare the log-prob of that last token.
                if len(window) < 2:
                    # There's no meaningful context or next token
                    # Could happen if pos=0, but here pos won't be 0 for the target portion
                    # We'll skip or treat it as log-prob=0
                    continue

                tokens_in = window[:-1]  # context
                token_label = window[-1] # the token we're measuring

                input_tensor = torch.tensor(tokens_in, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(input_tensor)  # shape: [1, seq_len, vocab_size]
                    # The last position's logits predict `token_label`
                    last_pos = logits[:, -1, :]  # [1, vocab_size]
                    log_probs = F.log_softmax(last_pos, dim=-1)  # [1, vocab_size]

                logprob_token = log_probs[0, token_label].item()
                total_logprob += logprob_token

                # Check if greedy
                predicted_token = torch.argmax(last_pos, dim=-1).item()
                if predicted_token != token_label:
                    is_greedy_overall = False

            outputs.append((total_logprob, int(is_greedy_overall)))

        return outputs

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float]]:
        """
        request.args = (text,)
        We compute the log-likelihood of the entire text,
        chunking so we never exceed 40 tokens in a single forward pass.

        Returns (ll,) for each request (a single float in a tuple).
        """
        results = []
        print("roolling: ", len(requests), "requests")
        for req in tqdm(requests):
            (text,) = req.args  # note the comma to unpack tuple
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < 2:
                # no pairs to predict
                results.append((0.0,))
                continue

            total_logprob = 0.0
            
            # We'll measure each token i in [1..end], predicted by i-1 (and preceding).
            for i in range(1, len(tokens)):
                # We take up to 39 tokens of context + 1 new token => 40 max
                window_start = max(0, i - self.max_seq_len + 1)
                window_end = i + 1
                window = tokens[window_start:window_end]  # up to 40 tokens

                # The last token in `window` is the one we're measuring
                if len(window) < 2:
                    continue
                tokens_in = window[:-1]
                token_label = window[-1]

                input_tensor = torch.tensor(tokens_in, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(input_tensor)   # [1, seq_len, vocab_size]
                    last_pos = logits[:, -1, :]        # [1, vocab_size]
                    log_probs = F.log_softmax(last_pos, dim=-1)

                total_logprob += log_probs[0, token_label].item()

            results.append((total_logprob,))

        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """
        request.args = (input_text, gen_kwargs_dict)
        We'll do generation with naive greedy approach, 
        ensuring we never exceed 40 tokens at once
        by truncating the left side if needed.
        Returns the *entire* prompt + newly generated text as a single string.
        """
        print("generate_until")
        outputs = []
        for req in tqdm(requests):
            prompt_str, gen_params = req.args

            # Extract generation parameters
            max_gen_toks = gen_params.get("max_gen_toks", 20)
            stop_sequences = gen_params.get("until", [])

            # Tokenize the prompt
            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            input_ids = torch.tensor([prompt_ids], device=self.device)

            # We'll generate up to max_gen_toks tokens, 
            # but we must ensure the input never exceeds 40 tokens
            generated_ids = input_ids
            for _ in range(max_gen_toks):
                current_len = generated_ids.size(1)
                if current_len >= self.max_seq_len:
                    # We can't add more tokens
                    break

                # Forward pass
                logits = self.model(generated_ids)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # If we want to handle "stop sequences" at the text level, decode each step:
                decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                # Check if any stop sequence is found
                stop_found = False
                for seq in stop_sequences:
                    if seq in decoded_text:
                        stop_found = True
                        break
                if stop_found:
                    break

                # If our prompt grows beyond 40, we would shift left
                # so the last 39 tokens remain as context. (But in practice,
                # we break anyway if >= self.max_seq_len.)
                # Example:
                if generated_ids.size(1) > self.max_seq_len:
                    # keep only last 39 tokens
                    generated_ids = generated_ids[:, -self.max_seq_len:]

            # Finally decode the entire input+output 
            final_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # If a stop sequence is found, cut off at first occurrence
            for seq in stop_sequences:
                idx = final_text.find(seq)
                if idx != -1:
                    final_text = final_text[:idx]
                    break

            outputs.append(final_text)

        return outputs

# --------------------------------------------------
# 1. Define the tasks you want to evaluate on
#    (use `tasks.get_task_dict()` if you want to
#    create a dict from a list of task names)
# --------------------------------------------------
task_list = [
    "piqa",       # Physical Interaction QA
    # "hellaswag",  # HellaSwag
    "boolq",
    "winogrande",
    "arc_easy",
    "arc_challenge",
     "openbookqa",
    #"wikitext",
]

#selected_tasks = tasks.get_task_dict(task_list)

# --------------------------------------------------
# 2. Set up Normal Transformer and Differential Transformer
# --------------------------------------------------

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
vocab_size = tokenizer.vocab_size

print("paaath: ", mod_path)

with (open(mod_path / "models/hyper_roberta.json", "r") as f):
    f = json.load(f)

seq_len = f["seq_len"]
f["vocab_size"] = vocab_size
print("Vocab size:", vocab_size)

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
# get the pretrained embedding
input_emb = roberta.get_input_embeddings().to(device)

# Freeze the embedding weights
for param in input_emb.parameters():
    param.requires_grad = False

output_emb = roberta.get_output_embeddings().to(device)
print(output_emb)
for param in output_emb.parameters():
    param.requires_grad = False

weight_path = mod_path / "models/wiki_training_transformer_roberta.pth"

weight_path_diff = mod_path / "models/wiki_training_differentialtransformer_roberta.pth"


model = LanguageModel_PretrainedEmb(input_emb, output_emb, vocab_size, embed_dim, hidden_dim, num_heads, head_dim, num_layers, l_init, max_seq_len=seq_len, device = device).to(device)
model_diff = LanguageModel_PretrainedEmb(input_emb, output_emb, vocab_size, embed_dim, hidden_dim, num_heads, head_dim, num_layers, l_init, diff=True, max_seq_len=seq_len, device = device).to(device)

if os.path.exists(weight_path):
    try :
        print(f"Loading weights from {weight_path}...")
        model.load_state_dict(torch.load(weight_path, map_location=torch.device(device), weights_only=True))
    except Exception as e:
        print(e)
        print("Couldn't upload the weights, maybe you change some hyperparameters ?")
        exit(e)
else:
    print(f"No weights file found at {weight_path} for transformer model. Starting from scratch.")

if os.path.exists(weight_path_diff):
    try :
        print(f"Loading weights from {weight_path_diff}...")
        model_diff.load_state_dict(torch.load(weight_path_diff, map_location=torch.device(device), weights_only=True))
    except Exception as e:
        print(e)
        print("Couldn't upload the weights, maybe you change some hyperparameters?")
        exit(e)

else:
    print(f"No weights file found at {weight_path_diff} for differential transformer model. Starting from scratch.")


test_model = TestLM(model, tokenizer, device)
test_model_diff = TestLM(model_diff, tokenizer, device)


if __name__ == "__main__":
    # --------------------------------------------------
    # 4. Run Evaluation for Model 1
    # --------------------------------------------------
    print("Evaluating Model 1...")
    results_model1 = evaluator.simple_evaluate(
        model=test_model,
        tasks=task_list,
        num_fewshot=0,     # Zero-shot, can set to other k for few-shot
        limit=None         # Limit # of evaluation examples (None = full)
    )

    # --------------------------------------------------
    # 5. Run Evaluation for Model 2
    # --------------------------------------------------

    print("Evaluating Model 2...")
    results_model2 = evaluator.simple_evaluate(
        model=test_model_diff,
        tasks=task_list,
        num_fewshot=0,
        limit=None
    )

    # --------------------------------------------------
    # 6. Print or Save Results
    # --------------------------------------------------
    print("\n============================")
    print("         RESULTS")
    print("============================")

    print("\nNormal Transformer Results:")
    for task_name, metrics in results_model1["results"].items():
        print(f"{task_name}: {metrics}")

    print("\nDifferential Transformer Results:")
    for task_name, metrics in results_model2["results"].items():
        print(f"{task_name}: {metrics}")

    # You can also compare aggregate metrics if desired
    #print("\nAggregate Scores:")
    #print("Model 1:", results_model1["aggregate"])
    #print("Model 2:", results_model2["aggregate"])

    # --------------------------------------------------
    # 7. Save Results to File
    # --------------------------------------------------

    # Save Normal Transformer results to JSON
    with open(mod_path / "results/results_transformer_0_shot.json", "w") as f:
        json.dump(results_model1["results"], f, indent=2)

    # Save Differential Transformer results to JSON
    with open(mod_path / "results/results_differential_transformer_0_shot.json", "w") as f:
        json.dump(results_model2["results"], f, indent=2)

