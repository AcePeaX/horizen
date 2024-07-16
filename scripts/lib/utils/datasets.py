import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import os


dir_path = os.path.dirname(os.path.realpath(__file__))

# setting path
sys.path.append(os.path.join(dir_path))
from tokenizer import CharTokenizer, END_CHAR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextChunksDataset(Dataset):
    def __init__(self, raw_data, context_length, tokenizer=None, device=device) -> None:
        super().__init__()
        self.device = device
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
            chunkTensor = self.tokenizer.encode(
                chunk, device=self.device
            )
            if(type(chunkTensor)!=torch.Tensor):
                chunkTensor = torch.tensor(chunkTensor, dtype=torch.long, device=device)
            if(len(chunkTensor)<context_length):
                chunkTensor = torch.cat([chunkTensor, torch.zeros(context_length-len(chunkTensor)+1, dtype=torch.long, device=device)])
            self.data.append(chunkTensor)
            for i in range(len(chunkTensor) - self.context_length - 1):
                self.mappingArray.append(idx)
                idx += 1
            self.mappingArray.append(idx)
            idx += self.context_length + 1
        self.mappingArray = torch.tensor(self.mappingArray, device=self.device)
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

    def __getitem__(self, index) -> torch.Tensor:
        if type(index) == int:
            index = index % len(self)
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
        elif type(index) == torch.Tensor:
            Lx = []
            Ly = []
            if len(index.shape) == 0:
                return self[index.item()]
            else:
                for k in index:
                    x, y = self[k.item()]
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


def get_batch(data: Dataset, batch_size: int, device=device):
    ix = torch.randint(len(data), (batch_size,), device=data.device)
    L = data[ix]
    x = torch.stack([x for x in L[0]])
    y = torch.stack([x for x in L[1]])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model,
    x_train: TextChunksDataset,
    x_val: TextChunksDataset,
    eval_iterations=50,
    batch_size=32,
):
    """Estimate the train and validation tests on a chunk of the dataset"""
    out = {}
    model.eval()
    for label, data in zip(["train", "val"], [x_train, x_val]):
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X, Y = get_batch(data, batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[label] = losses.mean()
    model.train()
    return out


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class ShardDataLoader:
    def __init__(self, B, T, dir, split, process_rank=1, num_processes=1, current_shard=0):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = dir
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if __name__=='__name__':
            print(f"found {len(shards)} shards for split {split}")
        self.reset(current_shard)

    def reset(self, current_shard=0):
        # state, init at shard zero
        self.current_shard = current_shard % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
