import torch
import torch.nn as nn
import torch.nn.functional as F


from .module import Module
from .utils import Block

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# setting path
sys.path.append(os.path.join(dir_path, ".."))

from utils.tokenizer import CharTokenizer
from utils.datasets import TextChunksDataset


class BasicSelfAttentionLanguageModel(Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_layers: int,
        context_size=None,
        n_heads=4,
        dropout=0.2,
    ):
        """
        If vocab_size is a Dataset with context_length, then no need to specify context_size
        """
        super().__init__()
        if context_size == None:
            if type(vocab_size) == TextChunksDataset:
                context_size = vocab_size.context_length
            else:
                raise Exception("You need to specify the context length")
        self.context_size = context_size
        if type(vocab_size) == TextChunksDataset:
            vocab_size = len(vocab_size.tokenizer)
        elif type(vocab_size) == CharTokenizer:
            vocab_size = len(vocab_size)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_heads = n_heads
        # each token has a probability distribution of appearing depending on the last token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(self.context_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads, self.context_size, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx)  # (B,T,C)
        pos_embd = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = tok_embd + pos_embd  # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

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
            logits, loss = self(idx[:, -self.context_size :])
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled text to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def save(self, path, epoch=None, optimizer_dict=None):
        torch.save({'model':self,'epoch':epoch,'optimizer_dict':optimizer_dict}, path)

    def load(self, path) -> list:
        dict = torch.load(path)
        m = dict['model']
        self.token_embedding_table = m.token_embedding_table
        self.position_embedding_table = m.position_embedding_table
        self.blocks = m.blocks
        self.ln_f = m.ln_f
        self.lm_head = m.lm_head
        return dict['epoch'], dict['optimizer_dict']
