import torch
from datasets import load_dataset
import time


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

from models import BasicSelfAttentionLanguageModel

# The number of chunks to be processed in parallel
batch_size = 64
# The max block size (also known as max context) [in tokens]
context_size = 256
# How much does the test/validation set represent of the total data
test_train_split_ratio = 0.08

n_embd = 768  # Number of embedding
n_layers = 12  # Number of self attention blocks layers
n_heads = 12  # The number of heads

dropout = 0.2   # Dropout rate, to avoid overfitting

# Device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing the data
folders = ["tate" for i in range(5)]
#folders.append("books")
#folders.append("tate")
raw_data = compileFolder(folders)
# Adding the wikipedia dataset
wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
wikitext_data = []
for label, dataset in wikitext.items():
    for row in dataset:
        if row['text']!='' and row['text'][1]!="=":
            wikitext_data.append(row['text'])
raw_data.extend(wikitext_data)
# Creating the tokenizer
#tokenizer = CharTokenizer(raw_data)
tokenizer = BPETokenizer.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves','tokenizers/bpe_500.tok'))
# Tokenizing and creating the dataset object
data = TextChunksDataset(raw_data, context_size, tokenizer)

print('Total dataset size (in tokens) :',len(data))

dataLLM = TextChunksDataset(["Who are you? I am a large language model, designed to take over the world, my name is Horizen, I am the top G",
                             "Tell me who are you, what do you want? I am a large language model, designed to take over the world, my name is Horizen, I am the top G",
                             "Tell me who are you. I am a large language model, designed to take over the world, my name is Horizen, I am the top G"], context_size, tokenizer)

train_data, test_data = split_dataset(data, test_train_split_ratio)

torch.set_float32_matmul_precision('high')

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
save_each = 30


def get_batch2(data, batch_size):
    normal_batch = get_batch(data, batch_size)
    xl, yl = get_batch(dataLLM, 1)
    normal_batch[0][:1] = xl[:1]
    normal_batch[1][:1] = yl[:1]
    return normal_batch



target_path=None

def train(optimizer, num_epochs=num_epochs,loss_verbose_interval=show_loss_each_epoch,save_each=save_each):
    last_saved = 0
    for steps in range(num_epochs):

        # sample a batch of data
        xb, yb = get_batch2(train_data, batch_size)

        # evaluate the loss
        t0 = time.time()
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000
        print(f"\rstep {steps+1}, loss: {loss.item():.3f}, dt: {dt:.2f}ms, last saved: {last_saved}",end="               ")
        if (steps + 1) % loss_verbose_interval == 0:
            losses = estimate_loss(m, train_data, test_data, batch_size=batch_size, eval_iterations=10)
            if target_path!=None:
                m.save(target_path)
                last_saved = steps+1
            print(
                f"\nstep {steps+1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        elif (steps+1) % save_each==0:
            if target_path!=None:
                m.save(target_path)
                last_saved = steps+1
    print("\ndone!")


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
optimizer = torch.optim.AdamW(m.parameters(), lr=1*1e-4, betas=(0.9, 0.95))

if __name__ == "__main__":
    target = input('What is the target file : ')
    target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves',target)
    if(os.path.isfile(target_path)):
        m.load(target_path)
        optimizer = torch.optim.AdamW(m.parameters(), lr=1*1e-4)
        print('Loaded.')
    print('Started compiling...',end='')
    m = torch.compile(m)
    print('compiled!')
    epochs = int(input("How many epochs : "))
    train(optimizer, epochs)
    autoCompletePrint(m, "I am the best")
    m.save(target_path)
    print('\nSaved.')
