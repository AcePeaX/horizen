import torch
import time
from tqdm import tqdm
import math
import numpy as np
import random
import csv


import sys
import os

dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../lib")
)
# setting path
sys.path.append(dir_path)


from utils.tokenizer import CharTokenizer, BPETokenizer, END_CHAR
from utils.datasets import load_tokens

from models import BasicSelfAttentionLanguageModel


# Dynamic loading function for training
def load_prompts(file_path):
    with open(file_path, "rb") as f:
        while True:
            # Read length
            length_data = f.read(2)  # uint16 is 2 bytes
            if not length_data:
                break
            length = np.frombuffer(length_data, dtype=np.uint16)[0]
            
            # Read score
            score_data = f.read(4)  # float32 is 4 bytes
            score = np.frombuffer(score_data, dtype=np.float32)[0]
            
            # Read tokens
            tokens_data = f.read(length * 2)  # uint16 tokens, each 2 bytes
            tokens = np.frombuffer(tokens_data, dtype=np.uint16)
            first_occurence = [i for i in range(len(tokens)) if tokens[i] == 58][0]
            tokens = torch.tensor(tokens, dtype=torch.long)

            
            yield float(score), tokens, first_occurence

def train_test_split(data, test_ratio=0.2, seed=None):
    """
    Splits the dataset into training and test sets.

    Parameters:
        data (list): The dataset.
        test_ratio (float): Proportion of data to be used for testing (default: 0.2).
        seed (int): Seed for reproducibility (default: None).

    Returns:
        train_data (list): List of (score, tokens) tuples for training.
        test_data (list): List of (score, tokens) tuples for testing.
    """
    # Ensure reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Read all data into a list
    all_data = data
    
    # Shuffle the data
    random.shuffle(all_data)
    
    # Split into train and test sets
    split_index = int(len(all_data) * (1 - test_ratio))
    train_data = all_data[:split_index]
    test_data = all_data[split_index:]
    
    return train_data, test_data


data = [(score, data, first_occ) for score, data, first_occ in load_prompts('assets/hr/tokenized_prompts.bin')]

train_data, test_data = train_test_split(data, 0.15, seed=44)

# The max block size (also known as max context) [in tokens]
context_size = 1024
# The number of chunks to be processed in parallel
token_number_per_batch = 524_288
memory_batch_size = 4

micro_step_acum = (token_number_per_batch//context_size//memory_batch_size)
batch_size = micro_step_acum*memory_batch_size


# Device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BPETokenizer.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../saves','tokenizers/fineweb-edu-1024.tok'))
# Tokenizing and creating the dataset object

n_embd = 1024  # Number of embedding
n_layers = 16  # Number of self attention blocks layers
n_heads = 16  # The number of heads

dropout = 0.2   # Dropout rate, to avoid overfitting

torch.set_float32_matmul_precision('high')

model = BasicSelfAttentionLanguageModel(
    len(tokenizer),
    n_embd,
    n_layers,
    context_size=context_size,
    n_heads=n_heads,
    dropout=dropout
)

target = "finetuned"
csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../logs',target+'.csv')
csv_fields = ['epoch','train_loss','train_loss_sqrt','val_loss', 'val_loss_sqrt']
if(os.path.isfile(csv_path)):
    with open(csv_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(csv_fields)
epochs, _ = model.load("saves/pretrained-hr.zen", map_location=device)


model.to(device)


# Freeze parameters in layer1
for param in model.parameters():
    param.requires_grad = False


for param in model.blocks[10].parameters():
    param.requires_grad = True

unfrozen_layers = [8, 9, 10, 11]
for i in unfrozen_layers:
    for param in model.blocks[i].parameters():
        param.requires_grad = True
for param in model.lm_head.parameters():
        param.requires_grad = True


num_epochs = 1
save_each = 1

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.7*1e-4, betas=(0.9, 0.95))

token_counts = [tokens.shape[0] for score, tokens, _ in train_data]

print("Total ammount of training tokens:",sum(token_counts))


divide_data = 2
divide_len = len(data)//divide_data

compile = False
if compile:
    model = torch.compile(model)

def train(optimizer: torch.optim.Optimizer, start_epoch=0,num_epochs=num_epochs,save_each=save_each):
    for steps in range(start_epoch,start_epoch+num_epochs):
        # evaluate the loss
        val_loss_accum = None
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        acum_loss = 0
        model.train()
        for micro_step in tqdm(range(len(train_data)), unit=' epoch', desc=f"Epoch {steps+1: 05d}/{start_epoch+num_epochs: 05d} : "):
            # sample a batch of data
            micro_data = train_data[micro_step:micro_step+1]
            xb, yb = [tokens[:-1] for _, tokens, _ in micro_data], [tokens[1:] for _, tokens, _ in micro_data]
            first_occ = micro_data[0][2]
            score = micro_data[0][0]
            xb = xb[0][None, :]
            yb = yb[0][None, :]
            if xb.shape[1]>model.position_embedding_table.weight.size()[0]:
                xb = xb[:,:1024]
                yb = yb[:,:1024]
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb, loss_offset=first_occ)
            loss = loss
            loss = (torch.exp(-loss)-score)
            loss = loss**2/len(train_data)
            acum_loss += loss.detach()
            loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        print("The train loss is:",acum_loss.item(),"| The train standard deviation is:",np.sqrt(acum_loss.item()))

        model.eval()
        val_loss_accum = 0
        for micro_step in tqdm(range(len(test_data)), unit=' epoch', desc=f"Epoch {steps+1: 05d}/{start_epoch+num_epochs: 05d} : "):
            # sample a batch of data
            micro_data = test_data[micro_step:micro_step+1]
            xb, yb = [tokens[:-1] for _, tokens, _ in micro_data], [tokens[1:] for _, tokens, _ in micro_data]
            first_occ = micro_data[0][2]
            score = micro_data[0][0]
            xb = xb[0][None, :]
            yb = yb[0][None, :]
            if xb.shape[1]>model.position_embedding_table.weight.size()[0]:
                xb = xb[:,:1024]
                yb = yb[:,:1024]
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                logits, loss = model(xb, yb, loss_offset=first_occ)
                loss = loss
                loss = (torch.exp(-loss)-score)
                loss = loss**2/len(test_data)
                val_loss_accum += loss.detach()
        print("The test loss is:",val_loss_accum.item(),"| The test standard deviation is:",np.sqrt(val_loss_accum.item()))
        with open(csv_path, 'a') as f:
            write = csv.writer(f)
            write.writerow([steps+1,acum_loss.item(),np.sqrt(acum_loss.item()),(None if val_loss_accum==None else val_loss_accum.item()),(None if val_loss_accum==None else np.sqrt(val_loss_accum.item()))])
        epochs = steps


target_path = "saves/pretrained-hr-2.zen"
        




train(optimizer, start_epoch=epochs, num_epochs=20)


model.save(target_path, epochs, None)