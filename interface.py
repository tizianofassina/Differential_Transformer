import torch

from pathlib import Path
import inspect
from language_model import LanguageModel_PretrainedEmb
import json
from transformers import RobertaTokenizer, RobertaForCausalLM

import gradio as gr


# Setting device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# The current directory
mod_path = Path(inspect.getfile(inspect.currentframe())).parent


# Introducing the tokenizer and embedding
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

roberta = RobertaForCausalLM.from_pretrained("roberta-base")
input_emb = roberta.get_input_embeddings().to(device)
output_emb = roberta.get_output_embeddings().to(device)

for param in input_emb.parameters(): # just to save graph calculation
    param.requires_grad = False
for param in output_emb.parameters():
    param.requires_grad = False


# uploading hyperparameters
with (open(mod_path / "models/hyper_roberta.json", "r") as f):
    f = json.load(f)

vocab_size = f["vocab_size"]
embed_dim = f["embed_dim"]
hidden_dim = f["hidden_dim"]
num_heads = f["num_heads"]
head_dim = f["head_dim"]
num_layers = f["num_layers"]
l_init = f["l_init"]
seq_len = f["seq_len"]

# Introducing the model
model = LanguageModel_PretrainedEmb(input_emb, output_emb, vocab_size, embed_dim, hidden_dim, num_heads, head_dim, num_layers, l_init, diff = True,  max_seq_len=seq_len, device = device).to(device)

#Uploading the weights. If the upload fails the program is closed.
weight_path = mod_path / "models/wiki_training_differentialtransformer_roberta.pth"

try:
    print(f"Loading weights from {weight_path}...")
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device), weights_only=True))
except FileNotFoundError:
    print(f"Weight file not found at {weight_path}.")
    exit()
except RuntimeError as e:
    print(f"Runtime error occurred: {e}. Please check your model architecture and weights.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

model.eval()

# Generation function
def generate_from_prompt(model, prompt_str, tokenizer, generated_length=20):
    """
        This function takes is the generation function : input a string, output a string.
        """
    # Tokenize the input prompt using RoBERTa tokenizer
    input_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors='pt').to(device)

    # Generate tokens from the model
    generated_ids = model.generate(input_ids, generated_length=generated_length)

    # Decode the generated token IDs back into text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def generate_text(input_string):
    """
        This function simply manage the interaction between the user and the model
        in the interface.
        """
    return generate_from_prompt(model, input_string, tokenizer, generated_length=30)


# The interface

css = """
.gradio-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background-color: #f4f1eb;
}

.gradio-interface {
    border-radius: 15px;
    padding: 20px;
    background-color: #ffffff;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    margin: 10px;
    width: 45%;
    max-width: 500px;
}

.gradio-title {
    font-size: 36px;  /* Increased title size */
    color: #4a4a4a;   /* Optional: change title color */
    text-align: center;
}

.gradio-button {
    background-color: #b8a18d;
    color: white;
    font-size: 16px;
    border-radius: 8px;
}

.gradio-button:hover {
    background-color: #9a8c7a;
}
"""


if __name__ == "__main__":

    interface = gr.Interface(
        fn=generate_text,
        inputs=gr.Textbox(label="Text to be completed", placeholder="Input text", lines=2),
        outputs=gr.Textbox(label="Text generated", lines=5),
        title="Differential Transformer Language Model",
        css=css
    )

    interface.launch()