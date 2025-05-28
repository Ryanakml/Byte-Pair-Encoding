# Import library yang diperlukan
import pickle
import sys

# Fungsi untuk memuat model BPE yang sudah dilatih
def load_model(model_path='pretrain/bpe_model.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            print(f"Model has {len(model.merges)} merges and {len(model.word_freqs)} unique words")
            return model
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        print("Please run pretrain/pretrain.py first to train and save the model.")
        sys.exit(1)

# Fungsi untuk menguji model dengan teks input
def test_tokenization(model, text=None):
    if text is None:
        # Teks default untuk testing
        text = "Love, hate, or feel meh about Harry Potter, it's hard to argue that J.K. Rowling filled the books with intentional writing choices."
    
    print(f"\nOriginal text:\n'{text}'")
    print(f"\nLength of original text: {len(text)} characters")
    
    try:
        tokens = model.tokenize(text)
        print(f"\nBPE tokenization result:")
        print(tokens)
        print(f"\nTotal tokens: {len(tokens)}")
        
        # Calculate compression ratio
        avg_token_length = sum(len(token) for token in tokens) / len(tokens) if tokens else 0
        compression_ratio = len(text) / len(tokens) if tokens else 0
        print(f"Average token length: {avg_token_length:.2f} characters")
        print(f"Compression ratio: {compression_ratio:.2f}")
        
        return tokens
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return []

# Test dengan beberapa contoh teks
def run_multiple_tests(model):
    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are fascinating topics.",
        "Python programming is fun and useful for data science.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*50}")
        print(f"TEST {i}")
        print(f"{'='*50}")
        test_tokenization(model, text)

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
        # Gunakan teks default dan beberapa test tambahan
        test_tokenization(model)
        
        # Ask if user wants to run more tests
        response = input("\nDo you want to run additional tests? (y/n): ").lower()
        if response == 'y':
            run_multiple_tests(model)
        
        # Interactive testing
        print("\nInteractive testing (type 'quit' to exit):")
        while True:
            user_input = input("\nEnter text to tokenize: ").strip()
            if user_input.lower() == 'quit':
                break
            if user_input:
                test_tokenization(model, user_input)

if __name__ == "__main__":
    main()