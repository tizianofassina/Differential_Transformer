import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import inspect
from datasets import load_dataset
import seaborn as sns

from evaluation import TestLM, test_model, test_model_diff

def evaluate_long_context(model: TestLM, texts):
    """
    Given a single (long) text string, computes:
      - nlls[i]: -log p( token[i] | tokens[:i], ... )
      - cum_avg_nll[i]: average of nlls up to position i

    Returns:
      nlls: list of per-token negative log-likelihoods
      cum_avg_nll: list of cumulative average NLLs
    """
    # Tokenize the input text (no special tokens)
    total_nlls = torch.zeros(40)
    total_cum_avg_nll = torch.zeros(40)
    for text in tqdm(texts):

        tokens = model.tokenizer.encode(text, add_special_tokens=False)
        tokens = tokens[0:40]
        
        nlls = torch.zeros(40)  # per-token NLL
        for i in range(1, len(tokens)):
            # We measure the log-prob of tokens[i] given tokens up to i-1
            # Cap the context window to model.max_seq_len
            window_start = max(0, i - model.max_seq_len + 1)
            window_end = i + 1  # slice is exclusive, so this includes token[i]
            window = tokens[window_start:window_end]

            # If there's fewer than 2 tokens, skip
            if len(window) < 2:
                # This can happen if i < model.max_seq_len but we prefer at least 2 tokens
                continue

            # All but the last token is the context
            tokens_in    = window[:-1]
            token_target = window[-1]

            # Run the model
            input_tensor = torch.tensor(tokens_in, device=model.device).unsqueeze(0)
            with torch.no_grad():
                logits = model.model(input_tensor)       # [1, seq_len, vocab_size]
                last_logits = logits[:, -1, :]          # [1, vocab_size] (logits for the final position)
                log_probs = F.log_softmax(last_logits, dim=-1)  # [1, vocab_size]

            # Extract log-prob of the true token
            log_prob = log_probs[0, token_target].item()
            nll = -log_prob
            nlls[i] = nll

        # Compute cumulative average NLL

        cum_avg_nll = torch.cumsum(nlls, dim=0) / torch.arange(1, len(nlls) + 1, dtype=torch.float)

        total_nlls += torch.tensor(nlls)
        total_cum_avg_nll += torch.tensor(cum_avg_nll)

    
    return total_nlls / len(texts), total_cum_avg_nll / len(texts)



def plot_cumulative_nll(cum_avg_nll):
    """
    Plots the cumulative average NLL against token positions.
    """
    positions = range(1, len(cum_avg_nll) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(positions, cum_avg_nll, label="Cumulative Avg NLL")
    plt.xlabel("Token position")
    plt.ylabel("Cumulative Avg NLL")
    plt.title("Cumulative Average NLL vs. Token Position")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    sns.set_theme("poster")

    # Load the test split of WikiText-2 (raw version)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Each entry in `dataset` has a field "text", which may sometimes be empty or just whitespace.
    # Let's filter out empty lines:
    texts = [item["text"] for item in dataset if item["text"].strip() != ""]

    # Now we have a list of text samples from WikiText-2 test set.
    print(f"Number of text samples: {len(texts)}")
    print("Example text:", texts[0])

    mod_path = Path(inspect.getfile(inspect.currentframe())).parent

    # Example usage:

    text_to_evaluate = (
        "Since 1990 the total number of hours flown annually by the GA sector has remained in the range 1 @.@ 25 1 @.@ 35 million , the dominant sector being traditional GA flying , which accounts for 0 @.@ 6 million per year . An overall increase in aircraft numbers combined with nil growth in hours flown has brought the annual average utilisation per aircraft down from 157 hours in 1984 to 103 hours in 2002 . The decline in asset utilisation has led to speculation that the economic health of the GA industry is weakening , though the lack of data on profitability makes this difficult to confirm ."
    )

    nlls_1, cum_avg_1 = evaluate_long_context(test_model, texts)
    nlls_2, cum_avg_2 = evaluate_long_context(test_model_diff, texts)

    positions_1 = range(1, len(cum_avg_1)+1)
    positions_2 = range(1, len(cum_avg_2)+1)

    plt.figure(figsize=(8,5))
    plt.plot(positions_1[:-1], cum_avg_1[:-1], label="Normal Attention")
    plt.plot(positions_2[:-1], cum_avg_2[:-1], label="Differential Attention")
    plt.xlabel("Token position")
    plt.ylabel("Negative log likelihood")
    plt.title("Cumulative Average Negative Log Likelihood comparison")
    plt.legend()
    plt.savefig("cumulative_NLL.png", dpi=300)
    plt.show()
