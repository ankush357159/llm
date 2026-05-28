import pytest
from embedding.SimpleTokenizer import SimpleTokenizer

def test_basic_tokenization():
    tokenizer = SimpleTokenizer()
    text = "Hello World"
    
    result = tokenizer.tokenize(text)
    
    assert result == ["hello", "world"]
    
def test_lowercase_conversion():
    tokenizer  = SimpleTokenizer()
    text = "CAPITALS"
    
    result = tokenizer.tokenize(text)
    
    assert result == ["capitals"]