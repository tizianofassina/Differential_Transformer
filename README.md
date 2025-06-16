# Differential Transformer - AMAL Project


This project aims to reproduce the architecture and results presented in the paper: **Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G., & Wei, F. (2024). Differential Transformer. arXiv:2410.05258**.

It was developed as part of the Advanced Machine Learning course in the **Master's DAC** program at **Sorbonne Université**.

The project implements two types of architectures:
 - A classical Transformer for language generation
 - A Differential Transformer, which differs from the classical version in its attention mechanism.

## Repository Structure

`model.py`
Implements the architectures of both the classical and differential Transformer models.

`language_model.py`
Uses the Transformer models to build a language model. Two versions were developed:
A model using a pretrained RoBERTa word embedding
A model with a trainable embedding

`normalization.py`
Contains various normalization functions used in the models.

`swiglu.py`
Implements the Swish function, as defined in this paper, and used in the Differential Transformer model.

`wiki_training.py`
Defines the training process for both the classical and differential Transformer using trainable embeddings on the WikiText-2-raw-v1 dataset.

`wiki_training_roberta.py`
Defines the training process for both Transformer models using pretrained RoBERTa embeddings.

`evaluation.py` & `context_eval.py`
Contain evaluation scripts for testing the trained models.

`interface.py`
Provides a simple interface for using the trained Differential Transformer language model.

`requirements.txt`
You can find the environment requirements in this file. 
## 🔗 Links  
- **Repository GitHub**: [https://github.com/psotom/AMAL.git](https://github.com/psotom/AMAL.git)  
- **Paper originale**: [arXiv:2410.05258](https://arxiv.org/abs/2410.05258)  

