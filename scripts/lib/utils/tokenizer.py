import torch
from tiktoken import Encoding
import regex as re

END_CHAR = "[S]"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharTokenizer:
    def __init__(self, text: str, addAlphabet=False) -> None:
        """
        Parameters:
        -------------
        text: str
            could be the entire test or the set of characters
        addAlphabet: bool
            is by default False. If True,
        """
        compiledText = text
        if type(text) == list:
            compiledText = " ".join(text)

        # it is mainly to add all letters in the alphabet
        if addAlphabet:
            compiledText += "".join(
                [chr(i) for i in range(ord("A"), ord("Z") + 1)]
            ) + "".join([chr(i) for i in range(ord("a"), ord("z") + 1)])

        temp = list(set(compiledText))
        temp.insert(0, END_CHAR)
        self.vocab = sorted(temp)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def __len__(self):
        return len(self.stoi)

    def encode(self, text, isMiddle=True, device=device) -> torch.Tensor:
        """
        Convert the text into tokens
        """
        L = []
        if not isMiddle:
            L = [self.stoi[c] for c in text] + [self.stoi[END_CHAR]]
        else:
            L = [self.stoi[c] for c in text]
        return torch.tensor(L, dtype=torch.long, device=device)

    def decode(self, L: list) -> list:
        """
        Convert tokens into text (list)

        Parameters:
        -------------
        L: list
            list of tokens
        """
        return [self.itos[i.item()] for i in L]

    def decodeText(self, L: list) -> str:
        """
        Convert tokens into text

        Parameters:
        -------------
        L: list
            list of tokens
        """

        def nullifySpecialChars(char):
            if char == END_CHAR:
                return ""
            return char

        return "".join([nullifySpecialChars(self.itos[i.item()]) for i in L])


# ________________________________________________________________________________________
def get_stats(ids):
    if len(ids) == 0:
        return
    counts = {}
    if type(ids[0]) == list:
        for chunk in ids:
            for pair in zip(chunk, chunk[1:]):
                counts[pair] = counts.get(pair, 0) + 1
    else:
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    if len(ids) == 0:
        return
    if type(ids[0]) == list:
        newIds = []
        for ids_chunk in ids:
            i = 0
            subNewIds = []
            while i < len(ids_chunk):
                if (
                    i < len(ids_chunk) - 1
                    and ids_chunk[i] == pair[0]
                    and ids_chunk[i + 1] == pair[1]
                ):
                    subNewIds.append(idx)
                    i += 2
                else:
                    subNewIds.append(ids_chunk[i])
                    i += 1
            newIds.append(subNewIds)
        return newIds
    else:
        newIds = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newIds.append(idx)
                i += 2
            else:
                newIds.append(ids[i])
                i += 1
        return newIds


class BPETokenizer:
    def __init__(self) -> None:
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.mergeable_ranks = {bytes([idx]): idx for idx in range(256)}
        self.tokenizer = Encoding(
            "custom-encoding",
            pat_str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=self.mergeable_ranks,
            special_tokens={},
        )

    def addMerges(self, text, num_merges=1, verbose=False):
        tokenizer_pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        if verbose:
            print("preparing....", end="")
        splitted_data = re.findall(tokenizer_pat, text)
        ids = []
        for chunk in splitted_data:
            ids.append(self.tokenizer.encode(chunk))
        if verbose:
            print("done")
        vocab_len = len(self.vocab)
        for i in range(num_merges):
            stats = get_stats(ids)
            top_pair = max(stats, key=stats.get)
            idx = vocab_len + i
            if verbose:
                print(f"{i+1}/{num_merges} merging {top_pair} into a new token {idx}")
            ids = merge(ids, top_pair, idx)
            self.mergeable_ranks[self.vocab[top_pair[0]] + self.vocab[top_pair[1]]] = (
                idx
            )
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
        self.tokenizer = self.tokenizer = Encoding(
            "custom-encoding",
            pat_str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=self.mergeable_ranks,
            special_tokens={},
        )

    def __len__(self):
        return len(self.vocab)

    def encode(self, text, device=None):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        if type(tokens)==torch.Tensor:
            tokens = [i.item() for i in tokens]
        return self.tokenizer.decode(tokens)

    def save(self, path: str):
        torch.save(self.vocab, path)

    @classmethod
    def load(cls, path: str):
        BPE_tok = cls()
        BPE_tok.vocab = torch.load(path)
        BPE_tok.mergeable_ranks = {byte: idx for idx, byte in BPE_tok.vocab.items()}
        BPE_tok.tokenizer = Encoding(
            "custom-encoding",
            pat_str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=BPE_tok.mergeable_ranks,
            special_tokens={},
        )
        return BPE_tok
