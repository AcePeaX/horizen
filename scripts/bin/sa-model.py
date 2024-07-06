import torch


import sys
import os

dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../lib")
)
# setting path
sys.path.append(dir_path)


from utils.compile import compileFolder
from utils.tokenizer import CharTokenizer, END_CHAR
from utils.datasets import TextChunksDataset, split_dataset, get_batch, estimate_loss

from transformers import BasicSelfAttentionLanguageModel

# The number of chunks to be processed in parallel
batch_size = 32
# The max block size (also known as max context) [in tokens]
block_size = 16
# How much does the test/validation set represent of the total data
test_train_split_ratio = 0.1

n_embd = 16  # Number of embedding
n_layers = 4  # Number of self attention blocks layers
context_size = 16  # Context length

head_size = 16  # The size of the heads (combiened)
n_heads = 4  # The number of heads

# Device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing the data
raw_data = compileFolder("tate")
# Creating the tokenizer
tokenizer = CharTokenizer(raw_data)
# Tokenizing and creating the dataset object
data = TextChunksDataset(raw_data, block_size, tokenizer)

train_data, test_data = split_dataset(data, test_train_split_ratio)


m = BasicSelfAttentionLanguageModel(train_data, n_embd)
m.to(device)
xb, yb = train_data[:10]
out = m(xb, yb)
print(
    tokenizer.decodeText(
        m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0]
    )
)

num_epochs = 1000
show_loss_each_epoch = 1000


def train(optimizer, num_epochs=num_epochs, loss_verbose_interval=show_loss_each_epoch):
    for steps in range(num_epochs):

        # sample a batch of data
        xb, yb = get_batch(train_data, batch_size)

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (steps + 1) % loss_verbose_interval == 0:
            losses = estimate_loss(m, train_data, test_data, batch_size=batch_size)
            print(
                f"step {steps+1}: train loss {losses ['train']:.4f}, val loss {losses ['val']:.4f}"
            )
    print("done!")


def autoCompletePrint(model, text, max_tokens=300, step=1):
    idx = tokenizer.encode(text)
    i = len(idx)
    print(text, end="")
    idx = torch.reshape(idx, (1, len(idx)))
    for j in range(0, max_tokens, step):
        res = m.generate(idx=idx, max_new_tokens=2)
        print(tokenizer.decodeText(res[0][i + j : i + j + step]), end="")
        idx = res


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)

if __name__ == "__main__":
    epochs = int(input("How many epochs : "))
    train(optimizer, epochs)
    autoCompletePrint("I am the best")