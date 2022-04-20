from typing import Tuple
from torch import nn


class MaxOverTimePooling(nn.Module):
    def forward(self, x):
        return nn.max


class CNNSentenceClassifier(nn.Module):
    def __init__(self, k_width: int, *h_windows: int):
        super().__init__()
        self.k_width = k_width
        self.embeddings = nn.Embedding(...)
        self.convolution = nn.ModuleList([
            nn.Conv1D(h_window) for h_window in h_windows
        ])
        self.max_over_time = nn.MaxPool1d(...)

    def forward(self, utterance):
        """
        Given an input sentence, x, a CNNSentenceClassifier convolves the
        word vectors into channels by context h_windows.
        """
        embedded = self.embedding(utterance)
        xs = self.convolution(embedded)

        
if __name__ == "__main__":
    "Hello, my name is Christian"
    CNNSentenceClassifier(3, 4, 5)