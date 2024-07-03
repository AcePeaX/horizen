import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.datasets import TextChunksDataset
from utils.tokenizer import CharTokenizer

from ..module import Module


class BigramLanguageBaseModel(Module):
    def __init__(self, vocab_size: int | CharTokenizer | TextChunksDataset):
        super().__init__()

        if type(vocab_size) == TextChunksDataset:
            vocab_size = len(vocab_size.tokenizer)
        elif type(vocab_size) == CharTokenizer:
            vocab_size = len(vocab_size)
        # each token has a probability distribution of appearing depending on the last token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled text to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
