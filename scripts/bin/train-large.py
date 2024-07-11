import torch
from datasets import load_dataset
import time
from tqdm import tqdm
import math
import csv

import sys
import os

dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../lib")
)
# setting path
sys.path.append(dir_path)


from utils.compile import compileFolder
from utils.tokenizer import CharTokenizer, BPETokenizer, END_CHAR
from utils.datasets import ShardDataLoader, split_dataset, get_batch, estimate_loss

from models import BasicSelfAttentionLanguageModel


# The max block size (also known as max context) [in tokens]
context_size = 1024
# The number of chunks to be processed in parallel
token_number_per_batch = 524_288
memory_batch_size = 8

micro_step_acum = (token_number_per_batch//context_size//memory_batch_size)
batch_size = micro_step_acum*memory_batch_size

# How much does the test/validation set represent of the total data
test_train_split_ratio = 0.08

n_embd = 768  # Number of embedding
n_layers = 12  # Number of self attention blocks layers
n_heads = 12  # The number of heads

dropout = 0.2   # Dropout rate, to avoid overfitting

# Device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing the data
dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../assets','fineweb-edu-score-2')
train_data = ShardDataLoader( memory_batch_size, context_size, dataset_dir, 'train')
test_data = ShardDataLoader( memory_batch_size, context_size, dataset_dir, 'val')
# Creating the tokenizer
#tokenizer = CharTokenizer(raw_data)
tokenizer = BPETokenizer.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves','tokenizers/fineweb-edu-1024.tok'))
# Tokenizing and creating the dataset object


torch.set_float32_matmul_precision('high')

m = BasicSelfAttentionLanguageModel(
    len(tokenizer),
    n_embd,
    n_layers,
    context_size=context_size,
    n_heads=n_heads,
    dropout=dropout
)
m.to(device)

num_epochs = 1000
show_loss_each_epoch = 200
save_each = 10
val_loss_frequency = 100
gen_frequency = 100


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


target_path = None
csv_path = None
csv_gen_path = None
csv_fields = ['epoch','train_loss','val_loss','lr']
csv_gen_fields = ['epoch','prompt','generated']

def train(optimizer: torch.optim.Optimizer, start_epoch=0,num_epochs=num_epochs,loss_verbose_interval=show_loss_each_epoch,save_each=save_each):
    last_saved = 0
    for steps in range(start_epoch,start_epoch+num_epochs):

        

        # evaluate the loss
        val_loss_accum = None
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        acum_loss = 0
        for micro_step in tqdm(range(micro_step_acum), unit=' epoch', desc=f"Epoch {steps+1: 05d}/{start_epoch+num_epochs: 05d} : "):
            # sample a batch of data
            xb, yb = train_data.next_batch()
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = m(xb, yb)
            loss = loss/micro_step_acum
            acum_loss += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        lr = get_lr(steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000
        print(f"\rstep {steps+1}, loss: {acum_loss:.3f} | lr: {lr: .4e} | dt: {dt:.2f}ms | last saved: {last_saved}",end="               ")
        # once in a while evaluate our validation loss
        if (steps+1) % val_loss_frequency == 0:
            m.eval()
            test_data.reset()
            val_loss_steps = 64
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = test_data.next_batch()
                    x, y = x.to(device), y.to(device)    
                    logits, loss = m(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            print(f"\nvalidation loss: {val_loss_accum.item():.4f}",end='')
        if (steps+1) % gen_frequency==0:
            prompt = "Hello, I am"
            idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
            idx = torch.reshape(idx, (1, len(idx)))
            result = tokenizer.decode(m.generate(idx, 200)[0])
            with open(csv_gen_path, 'a') as f:
                write = csv.writer(f)
                write.writerow([steps+1,prompt,result])
            print(f"step {steps+1}, generation:",result,end='')
        if (steps+1) % save_each==0:
            print('')
            if target_path!=None:
                m.save(target_path, steps, optimizer.state_dict())
                last_saved = steps+1
        with open(csv_path, 'a') as f:
            write = csv.writer(f)
            write.writerow([steps+1,acum_loss.item(),(None if val_loss_accum==None else val_loss_accum.item()),lr])
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
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../logs',target+'.csv')
    csv_gen_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../logs',target+'.gen.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    last_epoch = -1
    if(os.path.isfile(target_path)):
        last_epoch, optim_dict = m.load(target_path)
        optimizer = torch.optim.AdamW(m.parameters(), lr=1*1e-4)
        optimizer.load_state_dict(optim_dict)
        print('Loaded the model :',target,'with ',end='')
        print(sum(p.numel()for p in m. parameters())/1e6,'M parameters')
    else:
        print('Created the model with ',end='')
        print (sum(p.numel()for p in m. parameters())/1e6,'M parameters')
        with open(csv_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(csv_fields)
        with open(csv_gen_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(csv_gen_fields)
    print('Started compiling...',end='')
    m = torch.compile(m)
    print('compiled!')
    epochs = int(input("How many epochs : "))
    train(optimizer, num_epochs=epochs, start_epoch=(-1 if last_epoch==None else last_epoch)+1)
    autoCompletePrint(m, "I am the best")
    m.save(target_path, last_epoch+epochs, optimizer.state_dict())
    print('\nSaved.')
