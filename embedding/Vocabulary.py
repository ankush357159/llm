from embedding.SimpleTokenizer import SimpleTokenizer

"""
Manages a vocabulary mapping unique text tokens to numerical IDs and 
encodes text strings into lists of token IDs based on the learned vocabulary.
"""
class Vocabulary:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        
    def build(self, texts):
        vocab = set()
        tokenizer = SimpleTokenizer()
        
        for text in texts:
            tokens = tokenizer.tokenize(text)
            vocab.update(tokens)
            
        vocab = sorted(list(vocab))
        
        self.word_to_id = {
            word : idx for idx, word in enumerate(vocab)
        }
        
    
    def encode(self, text):
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(text)
        return [
            self.word_to_id[token] for token in tokens if token in self.word_to_id
        ]
        
    def vocab_size(self):
        return len(self.word_to_id)