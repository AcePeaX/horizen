import torch
import torch.nn as nn
import torch.nn.functional as F



class Head(nn.Module):
    """One head of self attention"""

    def __init__(
        self, n_embd: int, head_size: int, block_size: int, dropout=0.2
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)
        # Compute attention score ('affinities')
        #wei = (
        #    q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        #)  # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        #wei = F.softmax(wei, dim=-1)
        # Perform the weighted aggregation of the values
        #out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out  # (B, T, C)


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(
        self, num_heads: int, n_embd: int, head_size: int, block_size: int, dropout=0.2
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd: int, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_heads, block_size, dropout=0.2):
        # n_embd: embedding dimension, n_heads: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_heads
        self.debug_int = 0
        self.debug_order = 0
        self.sa = MultiHeadAttention(
            n_heads, n_embd, head_size, block_size, dropout=dropout
        )
        self.ffwd = FeedForward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        if(self.debug_int==5):
            print(self.debug_int,self.debug_order)
            print(x.shape)
        x = x + self.sa(self.ln1(x))

        if(self.debug_int==5):
            print(x.shape)
        x = x + self.ffwd(self.ln2(x))
        return x
