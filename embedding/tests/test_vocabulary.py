import  pytest
from embedding.Vocabulary import Vocabulary

def test_vocabulary():
    """Test that vocabulary builds unique sorted tokens from input texts."""
    vocab = Vocabulary()
    texts = ["Hello world", "world of NLP"]

    vocab.build(texts)

#     Expected unique tokens: "hello", "world", "of", "nlp"
    expected_vocab = {
        "hello": 0,
        "nlp": 1,
        "of": 2,
        "world": 3
    }

    assert vocab.vocab_size() == 4
    assert "hello" in vocab.word_to_id
    assert "nlp" in vocab.word_to_id
    assert vocab.word_to_id["hello"] == 0
    assert vocab.word_to_id["nlp"] == 1
    assert vocab.word_to_id == expected_vocab


def test_vocabulary_encode_unknown_words():
    """Test that unknown words are safely ignored during encoding."""
    vocab = Vocabulary()
    vocab.build(["hello", "world"])
    # 'python' is ignored because it was not in the training texts
    encoded = vocab.encode("Hello python world")
    assert encoded == [0, 1]


def test_vocabulary_empty_input():
    """Test encoding an empty string or text with no matching tokens."""
    vocab = Vocabulary()
    vocab.build(["Hello world"])

    assert vocab.encode("") == []
    assert vocab.encode("unknown words only") == []

def test_vocab_size_initially_zero():
    """Test that an unbuilt vocabulary has a size of zero."""
    vocab = Vocabulary()
    assert vocab.vocab_size() == 0

