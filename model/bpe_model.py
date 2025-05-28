## dependencies
from collections import Counter, defaultdict    # for counting frequencies
from transformers import AutoTokenizer          # for pre-tokenization similarly to BERT

class BPE():
    """ 
    This class encapsules the logic for training BPE tokenizer and use it for tokenize
    """

    def __init__(self, corpus, vocab_size):
        self.corpus = corpus            # list of raw text strings
        self.vocab_size = vocab_size    # target size of subword vocabulary

        # pre-tokenizer the corpus into wrods using BERT pre-tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # word_freqs : frequency of each word in the corpus
        self.word_freqs =  defaultdict(int)
        self.splits = {}     # how each word is split into sobwords is stored here
        self.merges = {}     # how each pair of subwords is merged is stored here

    def train(self):
        """Train BPE tokenizer."""
        
        # split corpus into words like bert does
        """
        words_with_offsets : words and its offsets(positions) in the original text

        text = "Hello world"
        words_with_offsets = pre_tokenize_str(text)

        [('Hello', (0, 5)), ('world', (6, 11))]

        • "Hello" appear at position 0 till 5
	    • "world" appear at position 6 till 11
        """

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
        
        # prepand a special token for each characters in the alphabet, this vocab will
        # contain all possible subwords in the corpus and will be updated during training for the
        # combination of characters
        """ "abc" prepended with "X" → "Xabc" """
        vocab = ['</w>'] + alphabet.copy()

        # split each word into individual characters before training
        """
        self.splits = {
        'hello': ['h', 'e', 'l', 'l', 'o'],
        'world': ['w', 'o', 'r', 'l', 'd']}
        """
        self.splits = {
            word: [c for c in word] for word in self.word_freqs.keys()
        }

        while len(vocab) < self.vocab_size:
            """
            Untill vocab reaches the desire size:
            1. Count the character pair frequencies
            2. Find the most frequent pair
            - pair_freqs = {('l', 'o'): 3, ('h', 'e'): 5, ('e', 'l'): 2}
            - best_pair = ('h', 'e')  # 5 (max freq = 5)
            3. Merge that pair in all words that have it
            4. Update Vocab and store the merge
            """

            # 1. Count the character pair frequencies
            pair_freqs = self.compute_pair_freqs()

            # Additional: check if there are any pair to merge
            if not pair_freqs:
                print("No pair to merge, vocab size is already reached")
                break

            # 2. Find the most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)

            # 3. Merge that pair in all words that have it
            self.splits = self.merge_pair(best_pair[0], best_pair[1])
            self.merges[best_pair] = best_pair[0] + best_pair[1]

            # 4. Update Vocab and store the merge
            """
            vocab = ['t', 'h']
            vocab.append(best_pair[0] + best_pair[1])
            vocab = ['t', 'h', 'he']
            """
            vocab.append(best_pair[0] + best_pair[1])

    
    def compute_pair_freqs(self):
        """
        for every word that already split into subwords.
        'hello' → ['h', 'e', 'l', 'l', 'o'])
        1. Counts how often each adjecent (berdampingan) character pair appears
        ['h', 'e', 'l', 'l', 'o']
        → pair: ('h','e'), ('e','l'), ('l','l'), ('l','o')
        2. Multiple by the word frequency in vocab
        """

        pair_freqs = defaultdict(int) # {"hello": 5, "world": 3}
        for word, freq in self.word_freqs.items():
            split = self.splits[word]       # "hello" → ['h', 'e', 'l', 'l', 'o']
            if len(split) == 1:             # skip if if its just one character
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
            
            """
            word_freqs = {
            • "hello" → ['h', 'e', 'l', 'l', 'o'] → freq = 2
            • "hell" → ['h', 'e', 'l', 'l'] → freq = 1
            }
            pair_freqs = {
            ('h', 'e'): 3,  # 2+1
            ('e', 'l'): 3,
            ('l', 'l'): 3,
            ('l', 'o'): 2
            }
            """
        return pair_freqs
        
    
    def merge_pair(self, a, b):
        """
        goes to all words already split into subwords and merges the pair, if its consecutive(same)
        1. Combines them into a+b only where they appear consecutively
        2. Update the split acocordingly
        'l' dan 'l' → consecutive
        split = ['h', 'e', 'l', 'l', 'o']
        new_split = ['h', 'e', 'll', 'o']
        """
        
        new_splits = {}
        
        for word in self.word_freqs:
            split = self.splits[word]       # one of words, splits["hello"] = ['h', 'e', 'l', 'l', 'o']`
            if len(split) == 1:             # helo = ['h', 'e', 'l', 'o']
                new_splits[word] = split    # adding single chars to new_splits
                continue

            new_split = []
            i = 0
            while i < len(split) - 1:
                # Check if there is a pair (a, b) at position i dan i+1
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    # if found pair a,b consecutive, combine them into individual token
                    # ['h', 'e', 'l', 'l', 'o']
                    # split[2] == 'l', split[3] == 'l' → concecutive → be 'll'
                    # Recent result: ['h', 'e', 'll']
                    new_split.append(a + b)
                    i += 2 # Skip next characterL cause already merged
                else:
                    new_split.append(split[i]) # append the character it self if not a pair (a, b)
                    i += 1 # Move to the next character
            
            new_splits[word] = new_split # update the split for the word

            """
            {
            "hello": ['h', 'e', 'll', 'o'],
            ...
            }
            """
        return new_splits

    
    def tokenize(self, text):
        """
        main function for tokenizing text wrapping all function together
        1. pre tokenizer word into tokens using bert pre-tokenizer by bert-base-uncased model
        - Input: "playing games"
        - Pre-tokenize: ["playing", "games"]
        2. split each token into individual characters
        - ["playing"] → ['p','l','a','y','i','n','g']
        - ["games"] → ['g','a','m','e','s']
        3. apply all learned merges in order
        - ['p','l','a','y','i','n','g'] 
        → ['pl','a','y','i','n','g'] 
        → ['pla','y','i','n','g']
        → ...
        4. Flatten the result and return
        - ['playing', 'games'] 
        → ['play', 'ing', 'games']
        """

        pre_tokenize_result = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pretokenized_text = [word for word, _ in pre_tokenize_result]   
        split_text = [[l for l in word] for word in pretokenized_text]  # ['p','l','a','y','i','n','g']

        for pair, marge in self.merges.items():
            for idx, split in enumerate(split_text):
                new_split = []
                i = 0
                while i < len(split):
                    if i< len(split) - 1 and split[i] == pair[0] and split[i+1] == pair[1]:
                        new_split.append(marge)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                split_text[idx] = split # ['pl','a','y','i','n','g']

        result = sum(split_text, [])
        """
        - ['pl', 'a', 'y', 'i', 'n', 'g']
        → ['pla', 'y', 'i', 'n', 'g']
        → ['playi', 'n', 'g']
        ......
        """
        return result