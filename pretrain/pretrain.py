# Import model BPE dari direktori model
from model.bpe_model import BPE
import pickle
import os

# Create necessary directories
os.makedirs('pretrain', exist_ok=True)

# Import data untuk melatih model
try:
    with open('data/data.txt', encoding="utf8") as f:
        corpus = f.readlines()
        # Strip whitespace and filter empty lines
        corpus = [line.strip() for line in corpus if line.strip()]
        print(f"Loaded corpus with {len(corpus)} lines")
        print(f"Sample: {corpus[:2]}")
except FileNotFoundError:
    print("Error: data/data.txt not found. Please create the data file first.")
    exit(1)

# Set hyperparameter ukuran vocabulary
vocab_size = 5000
print(f"Training BPE model with vocabulary size: {vocab_size}")

# Buat objek tokenizer BPE
MyBPE = BPE(corpus=corpus, vocab_size=vocab_size)

# Latih model
print("Starting training...")
MyBPE.train()
print("Training completed!")

# Print some statistics
print(f"Final vocabulary size: {len(MyBPE.merges) + len(set(''.join(MyBPE.word_freqs.keys())))}")
print(f"Number of merges learned: {len(MyBPE.merges)}")
print(f"Number of unique words: {len(MyBPE.word_freqs)}")

# Simpan model yang sudah dilatih
with open('pretrain/bpe_model.pkl', 'wb') as f:
    pickle.dump(MyBPE, f)
    print("Model saved to pretrain/bpe_model.pkl")