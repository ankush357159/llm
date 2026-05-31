import numpy as np


class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):

        self.embedding_matrix = (
            np.random.random((vocab_size, embedding_dim)) * 0.01
        )

    def forward(self, token_ids):
        return self.embedding_matrix[token_ids]