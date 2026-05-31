from collections.abc import Iterable

from embedding.SimpleTokenizer import SimpleTokenizer

"""
Manages a vocabulary mapping unique text tokens to numerical IDs and 
encodes text strings into lists of token IDs based on the learned vocabulary.
"""
class Vocabulary:
    def __init__(self) -> None:
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}

    def build(self, texts: Iterable[str]) -> None:
        vocab: set[str] = set()
        tokenizer = SimpleTokenizer()
        
        for text in texts:
            tokens = tokenizer.tokenize(text)
            vocab.update(tokens)
            
        vocab_words = sorted(vocab)

        self.word_to_id = {
            word : idx for idx, word in enumerate(vocab_words)
        }
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

    
    def encode(self, text: str) -> list[int]:
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(text)
        return [
            self.word_to_id[token] for token in tokens if token in self.word_to_id
        ]
        
    def vocab_size(self) -> int:
        return len(self.word_to_id)