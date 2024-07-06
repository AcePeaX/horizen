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

# The max block size (also known as max context) [in tokens]
context_size = 32
# Device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing the data
raw_data = compileFolder(["tate", "books"])
# Creating the tokenizer
tokenizer = CharTokenizer(raw_data)
# Tokenizing and creating the dataset object
data = TextChunksDataset(raw_data, context_size, tokenizer)



#_____________
target = 'sa-model-books.save'
#_____________

m = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves',target))
m.to(device)

autocomplete = ""

max_tokens = 200


autocomplete = input("Type in some text to autocomplete (STOP to stop) : ")
while autocomplete!='STOP':
    if autocomplete!='':
        idx = tokenizer.encode(autocomplete)
        idx = idx.reshape((1, len(idx)))
    else:
        idx = torch.zeros((1,1), dtype=torch.long, device=device)
    res = m.generate(idx=idx, max_new_tokens=max_tokens)
    for i in range(len(idx[0]),len(res[0])):
        if res[0,i].item()==0 and False:
            break
    print(tokenizer.decodeText(res[0,:i+1]))
    autocomplete = input("Type in some text to autocomplete (STOP to stop) : ")

