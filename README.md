# Byte-Pair Encoding (BPE) Tokenizer
A Python implementation of the Byte-Pair Encoding algorithm for subword-based tokenization from scratch. This project provides a clean implementation of BPE, which is a fundamental technique used in modern NLP models like GPT, BERT, and others.

## Overview
Byte-Pair Encoding is a data compression technique that iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte. In NLP, it's adapted to merge the most frequent character pairs in a corpus, creating a vocabulary of subword units that balances vocabulary size and token length.

## Features
- Pure Python implementation of BPE tokenization algorithm
- BERT-style pre-tokenization for initial word splitting
- Configurable vocabulary size
- Training on custom corpora
- Tokenization of new text using the trained model

## Requirements
`pip install transformers, collections`

## Usage

### Installation

First, install the required dependencies:

```bash
pip install transformers
```
Second, run the pretrain_bpe_model.py file to pretrain the BPE model on the sample data.

```bash
python pretrain_bpe_model.py
```

### Training a BPE Model
```python
from bpe_model import BPE

# Load your corpus
with open('your_corpus.txt', encoding="utf8") as f:
    corpus = f.readlines()

# Set the desired vocabulary size
vocab_size = 1000

# Create and train the BPE tokenizer
tokenizer = BPE(corpus=corpus, vocab_size=vocab_size)
tokenizer.train()

# Tokenize new text
text = "Hello world!"
tokens = tokenizer.tokenize(text)
print(tokens)
```

Example Output:

'Love, hate, or feel meh about Harry Potter, it's hard to argue that J.K. Rowling filled the books with intentional writing choices. From made up words to the meanings of names to the well-scripted first and last lines of each novel, Rowling wanted to the writing to match the intricate fantasy world she created for the now-iconic boy wizard. To examine a few of these choices, I'll be taking a closer look at the first line of Harry Potter, as well as the last lines, from all of the Harry Potter novels.'

Tokenized Output:

['L', 'ov', 'e', ',', 'h', 'ate', ',', 'or', 'fe', 'el', 'me', 'h', 'about', 'H', 'ar', 'ry', 'P', 'ot', 'ter', ',', 'it', ''', 's', 'h', 'ard', 'to', 'ar', 'g', 'ue', 'that', 'J', '.', 'K', '.', 'R', 'ow', 'l', 'ing', 'f', 'ill', 'ed', 'the', 'bo', 'ok', 's', 'with', 'int', 'ent', 'ional', 'writ', 'ing', 'cho', 'ic', 'es', '.', 'F', 'rom', 'made', 'up', 'w', 'ord', 's', 'to', 'the', 'me', 'an', 'ing', 's', 'of', 'n', 'ames', 'to', 'the', 'well', '-', 'sc', 'ri', 'pt', 'ed', 'first', 'and', 'l', 'ast', 'l', 'in', 'es', 'of', 'e', 'ach', 'n', 'ov', 'el', ',', 'R', 'ow', 'l', 'ing', 'w', 'ant', 'ed', 'to', 'the', 'writ', 'ing', 'to', 'm', 'at', 'ch', 'the', 'in', 'tr', 'ic', 'ate', 'f', 'ant', 'as', 'y', 'w', 'orld', 'she', 'cre', 'ated', 'for', 'the', 'n', 'ow', '-', 'ic', 'on', 'ic', 'bo', 'y', 'w', 'iz', 'ard', '.', 'T', 'o', 'ex', 'am', 'ine', 'a', 'f', 'ew', 'of', 'the', 'se', 'cho', 'ic', 'es', ',', 'I', ''', 'l', 'l', 'be', 't', 'ak', 'ing', 'a', 'c', 'lo', 'ser', 'lo', 'ok', 'at', 'the', 'first', 'l', 'ine', 'of', 'H', 'ar', 'ry', 'P', 'ot', 'ter', ',', 'as', 'well', 'as', 'the', 'l', 'ast', 'l', 'in', 'es', ',', 'from', 'all', 'of', 'the', 'H', 'ar', 'ry', 'P', 'ot', 'ter', 'n', 'ov', 'el', 's', '.']  

## How It Works
1. Pre-tokenization : The text is first split into words using BERT's pre-tokenizer
2. Character-level splitting : Each word is split into individual characters
3. Pair frequency counting : The algorithm counts the frequency of each adjacent character pair
4. Merging : The most frequent pair is merged into a new token
5. Iteration : Steps 3-4 are repeated until the desired vocabulary size is reached

## Hyperparameters
- Vocabulary size : The target size of the subword vocabulary (base characters + learned merges)

## Training Data
For training, you can use:

- The included sample data
- Wikipedia corpus 
- Any custom text corpus

## Applications
- Text preprocessing for NLP models
- Understanding tokenization in transformer models
- Educational purposes for learning about subword tokenization

## License
MIT License