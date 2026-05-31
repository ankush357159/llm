import re

"""
A simple text tokenizer that converts text to lowercase and extracts 
alphanumeric words as tokens, ignoring punctuation.
"""
class SimpleTokenizer:
    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens