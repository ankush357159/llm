
import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.float64]

class PositionalEncoding:
    def __init__(self, max_len: int, embedding_dim: int):
        self.encoding: Array = np.zeros((max_len, embedding_dim))

        for pos in range(max_len):
            for i in range(0, embedding_dim, 2):
                angle = pos / np.power(10000, 2 * (i // 2) / embedding_dim)
                self.encoding[pos, i] = np.sin(angle)

                if i + 1 < embedding_dim:
                    self.encoding[pos, i + 1] = np.cos(angle)

    def forward(self, x: Array) -> Array:
        seq_len = x.shape[-2]
        return x + self.encoding[:seq_len]
