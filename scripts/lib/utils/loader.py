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

    def __getitem__(self, index, block_size=1) -> torch.Tensor:
        if type(index) == int:
            return (
                self.data[
                    self.mappingArray[index] : self.mappingArray[index]
                    + self.context_length
                ],
                self.data[self.mappingArray[index] + self.context_length],
            )
        elif type(index) == slice:
            L = []
            for k in range(index.start or 0, index.stop or len(self), index.step or 1):
                L.append(self[k])
            return L
