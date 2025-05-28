# Import model BPE dari direktori model
from model.bpe_model import BPE
import pickle

# Import data untuk melatih model
with open('data/data.txt', encoding="utf8") as f:  # Ubah path ke direktori data baru
    corpus = f.readlines()
    print(f"Loaded corpus with {len(corpus)} lines")
    print(f"Sample: {corpus[:2]}")

# Set hyperparameter ukuran vocabulary
vocab_size = 1000
print(f"Training BPE model with vocabulary size: {vocab_size}")

# Buat objek tokenizer BPE
MyBPE = BPE(corpus=corpus, vocab_size=vocab_size)

# Latih model
MyBPE.train()

# Simpan model yang sudah dilatih
with open('pretrain/bpe_model.pkl', 'wb') as f:
    pickle.dump(MyBPE, f)
    print("Model saved to pretrain/bpe_model.pkl")