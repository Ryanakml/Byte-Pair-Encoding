# Import library yang diperlukan
import pickle
import sys

# Fungsi untuk memuat model BPE yang sudah dilatih
def load_model(model_path='pretrain/bpe_model.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return model
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        print("Please run pretrain/pretrain.py first to train and save the model.")
        sys.exit(1)

# Fungsi untuk menguji model dengan teks input
def test_tokenization(model, text=None):
    if text is None:
        # Teks default untuk testing
        text = "Love, hate, or feel meh about Harry Potter, it's hard to argue that J.K. Rowling filled the books with intentional writing choices. From made up words to the meanings of names to the well-scripted first and last lines of each novel, Rowling wanted to the writing to match the intricate fantasy world she created for the now-iconic boy wizard."
    
    print(f"\nBPE tokenization result of text:\n'{text}'")
    tokens = model.tokenize(text)
    print(tokens)
    print(f"\nTotal tokens: {len(tokens)}")
    return tokens

# Main function
def main():
    # Load model
    model = load_model()
    
    # Test dengan teks default atau teks dari argumen command line
    if len(sys.argv) > 1:
        # Jika ada argumen, gunakan sebagai teks input
        input_text = ' '.join(sys.argv[1:])
        test_tokenization(model, input_text)
    else:
        # Gunakan teks default
        test_tokenization(model)

if __name__ == "__main__":
    main()