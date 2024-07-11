import torch


import sys
import os

dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../lib")
)
# setting path
sys.path.append(dir_path)


from utils.compile import compileFolder
from utils.tokenizer import CharTokenizer, END_CHAR, BPETokenizer
from utils.datasets import TextChunksDataset, split_dataset, get_batch, estimate_loss

from models import BasicSelfAttentionLanguageModel

# The max block size (also known as max context) [in tokens]
context_size = 32
# Device (gpu or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing the data
#raw_data = compileFolder(["tate", "books"])
# Creating the tokenizer
tokenizer = BPETokenizer.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves','tokenizers/fineweb-edu-1024.tok'))
# Tokenizing and creating the dataset object
#data = TextChunksDataset(raw_data, context_size, tokenizer)



#_____________
target = 'wiki-powered-sa-xl.save'
#_____________

target = ''
if target=='':
    target = input('What is the target model : ')

m = torch.load(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves',target)))
m.to(device)


autocomplete = ""

max_tokens = 200

print('Loaded the model :',target,'with ',end='')
print (sum(p.numel()for p in m. parameters())/1e6,'M parameters')

autocomplete = input("Type in some text to autocomplete (STOP to stop) : ")
while autocomplete!='STOP':
    if autocomplete=='RELOAD':
        print('\nReloading the model...',end='')
        m = torch.load(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves',target)))
        print('loaded :',target,'!')
    else:
        if autocomplete!='':
            idx = torch.tensor(tokenizer.encode(autocomplete), dtype=torch.long, device=device)
            idx = idx.reshape((1, len(idx)))
        else:
            idx = torch.zeros((1,1), dtype=torch.long, device=device)
        res = m.generate(idx=idx, max_new_tokens=max_tokens)
        for i in range(len(idx[0]),len(res[0])):
            if res[0,i].item()==0 and False:
                break
        print(tokenizer.decode(res[0,:i+1])+"\n")
    autocomplete = input("Type in some text to autocomplete (STOP to stop) : ")

