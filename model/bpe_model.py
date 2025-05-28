## dependencies
from collections import Counter, defaultdict
from transformers import AutoTokenizer # for pre-tokenization similarly to BERT

class BPE():
    """ 
    This class encapsules the logic for training BPE tokenizer and use it for tokenize
    """

    def __init__(self, corpus, vocab_size):
        self.corpus = corpus            # list of raw text strings
        self.vocab_size = vocab_size    # target size of subword vocabulary

        # pre-tokenizer the corpus into words using BERT pre-tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # word_freqs : frequency of each word in the corpus
        self.word_freqs = defaultdict(int)
        self.splits = {}     # how each word is split into subwords is stored here
        self.merges = {}     # how each pair of subwords is merged is stored here

    def train(self):
        """Train BPE tokenizer."""
        
        # compute the frequency of each word in the corpus
        for text in self.corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, _ in words_with_offsets]
            
            # Update a frequency dictionary for all words.
            for word in new_words:
                self.word_freqs[word] += 1

        # Base vocab for all possible characters in the corpus
        alphabet = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()
        
        # Initialize vocab with special token and alphabet
        vocab = ['</w>'] + alphabet.copy()

        # split each word into individual characters before training
        self.splits = {
            word: [c for c in word] for word in self.word_freqs.keys()
        }

        while len(vocab) < self.vocab_size:
            # 1. Count the character pair frequencies
            pair_freqs = self.compute_pair_freqs()
            
            # Check if we have any pairs to merge
            if not pair_freqs:
                break

            # 2. Find the most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)

            # 3. Merge that pair in all words that have it
            self.splits = self.merge_pair(best_pair[0], best_pair[1])
            self.merges[best_pair] = best_pair[0] + best_pair[1]

            # 4. Update Vocab and store the merge
            vocab.append(best_pair[0] + best_pair[1])

    def compute_pair_freqs(self):
        """
        for every word that already split into subwords.
        1. Counts how often each adjacent character pair appears
        2. Multiple by the word frequency in vocab
        """

        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        
        return pair_freqs
        
    def merge_pair(self, a, b):
        """
        goes to all words already split into subwords and merges the pair, if its consecutive
        """

        new_splits = {}
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                new_splits[word] = split
                continue

            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    new_split.append(a + b)
                    i += 2  # Skip the next token since we merged it
                else:
                    new_split.append(split[i])
                    i += 1
            
            new_splits[word] = new_split
        
        return new_splits

    def tokenize(self, text):
        """
        main function for tokenizing text wrapping all function together
        """

        pre_tokenize_result = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pretokenized_text = [word for word, _ in pre_tokenize_result]   
        split_text = [[l for l in word] for word in pretokenized_text]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(split_text):
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                        new_split.append(merge)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                split_text[idx] = new_split

        result = sum(split_text, [])
        return result