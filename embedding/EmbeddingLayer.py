import numpy as np
import numpy.typing as npt


IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float64]


class EmbeddingLayer:
    def __init__(self, vocab_size: int, embedding_dim: int):

        self.embedding_matrix: FloatArray = (
            np.random.random((vocab_size, embedding_dim)) * 0.01
        )

    def forward(self, token_ids: list[int] | IntArray) -> FloatArray:
        return self.embedding_matrix[token_ids]