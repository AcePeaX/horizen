import torch


import sys
import os

dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../lib")
)
# setting path
sys.path.append(dir_path)


from utils.compile import compileFolder
from utils.tokenizer import CharTokenizer, BPETokenizer, END_CHAR
from utils.datasets import TextChunksDataset, split_dataset, get_batch, estimate_loss

from transformers import BasicSelfAttentionLanguageModel

# The number of chunks to be processed in parallel
batch_size = 64
# The max block size (also known as max context) [in tokens]
context_size = 256
# How much does the test/validation set represent of the total data
test_train_split_ratio = 0.1

n_embd = 512  # Number of embedding
n_layers = 7  # Number of self attention blocks layers
n_heads = 8  # The number of heads

dropout = 0.2   # Dropout rate, to avoid overfitting

# Device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing the data
raw_data = compileFolder(["tate"])
# Creating the tokenizer
#tokenizer = CharTokenizer(raw_data)
tokenizer = BPETokenizer.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves','tokenizers/bpe_500.tok'))
# Tokenizing and creating the dataset object
data = TextChunksDataset(raw_data, context_size, tokenizer)

train_data, test_data = split_dataset(data, test_train_split_ratio)


m = BasicSelfAttentionLanguageModel(
    train_data,
    n_embd,
    n_layers,
    context_size=context_size,
    n_heads=n_heads,
    dropout=dropout
)
m.to(device)

num_epochs = 1000
show_loss_each_epoch = 200


target_path=None

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
            losses = estimate_loss(m, train_data, test_data, batch_size=batch_size, eval_iterations=10)
            if target_path!=None:
                m.save(target_path)
            print(
                f"step {steps+1}: train loss {losses ['train']:.4f}, val loss {losses ['val']:.4f}"
            )
    print("done!")


def autoCompletePrint(model, text, max_tokens=200, step=1):
    idx = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    i = len(idx)
    print(text, end="")
    idx = torch.reshape(idx, (1, len(idx)))
    for j in range(0, max_tokens, step):
        res = m.generate(idx=idx, max_new_tokens=2)
        print(tokenizer.decode(res[0][i + j : i + j + step]), end="")
        idx = res
    print('')


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=3*1e-4)

if __name__ == "__main__":
    target = input('What is the target file : ')
    target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves',target)
    if(os.path.isfile(target_path)):
        m.load(target_path)
        optimizer = torch.optim.AdamW(m.parameters(), lr=3*1e-4)
        print('Loaded.')
    epochs = int(input("How many epochs : "))
    train(optimizer, epochs)
    autoCompletePrint(m, "I am the best")
    m.save(target_path)
    print('\nSaved.')
