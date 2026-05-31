import numpy as np

from embedding.SelfAttention import SelfAttention
from embedding.SimpleTokenizer import SimpleTokenizer
from embedding.Vocabulary import Vocabulary
from embedding.EmbeddingLayer import EmbeddingLayer
from embedding.PositionalEncoding import PositionalEncoding
from embedding.LayerNorm import LayerNorm

from embedding.LayerNorm import LayerNorm


def main():
    print("Starting LLM Embedding Pipeline Execution...")


def run():
    print("\n================== 1. DATA SEEDING & TOKENS ==================")
    # Raw text used to train our mini-vocabulary system
    training_data = [
        "Hello world",
        "This is clean code",
        "Welcome to the world of NLP"
    ]

    tokenizer = SimpleTokenizer()
    sample_sentence = "Hello clean NLP world"
    tokens = tokenizer.tokenize(sample_sentence)
    print(f"Sample Input Sentence: '{sample_sentence}'")
    print(f"Tokens Extracted: {tokens}")

    print("\n================== 2. BUILDING VOCABULARY ==================")
    vocab = Vocabulary()
    vocab.build(training_data)
    print(f"Vocabulary Size Learned: {vocab.vocab_size()}")
    print(f"Word-to-ID Mapping Table:\n{vocab.word_to_id}")

    print("\n================== 3. TEXT ENCODING (STRING TO IDS) ==================")
    # Convert our sample string into integers
    token_ids = vocab.encode(sample_sentence)
    print(f"Encoded Token IDs Array: {token_ids}")

    print("\n================== 4. EMBEDDING LOOKUP (C-LEVEL INDEXING) ==================")
    # Initialize our embedding layer (Rows = Vocabulary Size, Columns = 4 Dimensions)
    embedding_dim = 4
    embedding_layer = EmbeddingLayer(vocab.vocab_size(), embedding_dim)

    # Get standard semantic embeddings
    word_embeddings = embedding_layer.forward(token_ids)
    print(f"Word Embeddings Shape: {word_embeddings.shape} (Tokens, Vector Dimension)")
    print("Raw Initialized Semantic Embeddings Matrix:")
    print(word_embeddings)

    print("\n================== 5. INJECTING POSITION SIGNALS ==================")
    # Set up Positional Encodings (Max Length = 10 tokens, Matching 4 Dimensions)
    max_sequence_length = 10
    pos_encoder = PositionalEncoding(max_sequence_length, embedding_dim)

    # Add coordinates directly to word embeddings
    final_contextual_vectors = pos_encoder.forward(word_embeddings)
    print(f"Final Output Vectors Shape: {final_contextual_vectors.shape}")
    print("Final Model Input Vectors (Semantic Embeddings + Position Coordinates):")
    print(final_contextual_vectors)

    print("\n================== 6. APPLYING LAYER NORMALIZATION ==================")
    layer_norm = LayerNorm(embedding_dim)
    print("Applying Layer Normalization to stabilize the output vectors...")
    normalized_vectors = layer_norm.forward(final_contextual_vectors)
    print(f"Normalized Vectors Shape: {normalized_vectors.shape}")
    print("Final Stabilized Layer Output Matrix (Mean=0, Var=1 per row):")
    print(normalized_vectors)

    print("\n================== 7. PROCESSING SELF-ATTENTION VALUES ==================")
    # Initialize attention layer using the pipeline's configured embedding dimensions
    attention_layer = SelfAttention(embedding_dim)

    # Run forward pass extracting diagnostic internal states
    attn_output, Q, K, V, scores, weights = attention_layer.forward(normalized_vectors)

    print(f"Tokens Monitored: {tokens}")
    print(f"\nQuery Matrix (Q) Shape: {Q.shape}")
    print(Q)
    print(f"\nKey Matrix (K) Shape: {K.shape}")
    print(K)
    print(f"\nValue Matrix (V) Shape: {V.shape}")
    print(V)

    print(f"\nAttention Scores Matrix (Raw dot-product scaled by 1/sqrt(d)) Shape: {scores.shape}")
    print(scores)

    print(f"\nAttention Weights Matrix (Softmax normalized probabilities across rows) Shape: {weights.shape}")
    # Display weights mapped cleanly alongside token labels for visualization
    print(f"{'Token':<10} | " + " ".join([f"{t:<8}" for t in tokens]))
    print("-" * 50)
    for i, row in enumerate(weights):
        row_str = " ".join([f"{val:.4f}  " for val in row])
        print(f"{tokens[i]:<10} | {row_str}")

    print(f"\nFinal Attention Output Context Vectors Shape: {attn_output.shape}")
    print(attn_output)


if __name__ == "__main__":
    main()
    run()
