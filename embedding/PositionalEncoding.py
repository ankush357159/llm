
import numpy as np

class PositionalEncoding:
    def __init__(self, max_len, embedding_dim):
        self.encoding = np.zeros((max_len, embedding_dim))

        for pos in range(max_len):
            for i in range(0, embedding_dim, 2):
                angle = pos / np.power(10000, 2 * (i // 2) / embedding_dim)
                self.encoding[pos, i] = np.sin(angle)

                if i + 1 < embedding_dim:
                    self.encoding[pos, i + 1] = np.cos(angle)

    def forward(self, x):
        seq_len = x.shape[-2]
        return x + self.encoding[:seq_len]
