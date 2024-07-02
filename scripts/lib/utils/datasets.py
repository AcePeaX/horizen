import torch
from torch.utils.data import Dataset
import sys
import os


dir_path = os.path.dirname(os.path.realpath(__file__))

# setting path
sys.path.append(os.path.join(dir_path))
from tokenizer import CharTokenizer, END_CHAR


class TextChunksDataset(Dataset):
    def __init__(self, raw_data, context_length, tokenizer=None) -> None:
        super().__init__()
        self.data = []
        if tokenizer == None:
            tokenizer = CharTokenizer(raw_data)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.mappingArray = []
        idx = 0
        if type(raw_data) == str:
            raw_data = [raw_data]
        for chunk in raw_data:
            chunkTensor = self.tokenizer.encode([END_CHAR] + list(chunk), False)
            self.data.append(chunkTensor)
            for i in range(len(chunkTensor) - self.context_length - 1):
                self.mappingArray.append(idx)
                idx += 1
            self.mappingArray.append(idx)
            idx += self.context_length + 1
        self.mappingArray = torch.tensor(self.mappingArray)
        self.data = torch.cat(self.data)

    def __len__(self) -> int:
        return len(self.mappingArray)

    def __str__(self):
        return (
            "TextChunksDataset(length: "
            + str(len(self))
            + ",context_length: "
            + str(self.context_length)
            + ")"
        )

    def __getitem__(self, index, block_size=1) -> torch.Tensor:
        if type(index) == int:
            return (
                self.data[
                    self.mappingArray[index] : self.mappingArray[index]
                    + self.context_length
                ],
                self.data[
                    self.mappingArray[index]
                    + 1 : self.mappingArray[index]
                    + self.context_length
                    + 1
                ],
            )
        elif type(index) == slice:
            Lx = []
            Ly = []
            for k in range(index.start or 0, index.stop or len(self), index.step or 1):
                x, y = self[k]
                Lx.append(x)
                Ly.append(y)
            return torch.stack(Lx), torch.stack(Ly)


def split_dataset(data, ratio):
    """
    Returns (train,test)
    """
    if type(data) == list:
        data = torch.cat(data)
        n = int(len(data) * ratio)
        return data[n:], data[:n]
    elif type(data) == TextChunksDataset:
        n = int(len(data) * ratio)
        train_data = TextChunksDataset("", data.context_length)
        test_data = TextChunksDataset("", data.context_length)
        train_data.data = data.data
        train_data.tokenizer = data.tokenizer
        test_data.data = data.data
        test_data.tokenizer = data.tokenizer
        train_data.mappingArray = data.mappingArray[n:]
        test_data.mappingArray = data.mappingArray[:n]
        return train_data, test_data
