import torch

END_CHAR = "[S]"


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
        temp.append(END_CHAR)
        self.vocab = sorted(temp)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text, isMiddle=True) -> torch.Tensor:
        """
        Convert the text into tokens
        """
        L = []
        if not isMiddle:
            L = [self.stoi[c] for c in text] + [self.stoi[END_CHAR]]
        else:
            L = [self.stoi[c] for c in text]
        return torch.tensor(L, dtype=torch.long)

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
