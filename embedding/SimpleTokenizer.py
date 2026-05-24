import numpy as np
import re

class SimpleTokenizer:
    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens