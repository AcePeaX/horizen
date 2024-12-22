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

m = torch.load(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves',target)))['model']
m.to(device)

if m.vocab_size==1025:
    tokenizer = BPETokenizer.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves','tokenizers/fineweb-edu-1025.tok'))


autocomplete = ""

max_tokens = 200

print('\033[0mLoaded the model :',target,'with ',end='')
print (sum(p.numel()for p in m. parameters())/1e6,'M parameters')

res = torch.zeros((1,1), dtype=torch.long, device=device)
prompt_end = 0
res[0,0]=tokenizer.eot_token
autocomplete = input("Type in some text to autocomplete (STOP to stop) : ")
while autocomplete!='STOP':
    if autocomplete=='RELOAD':
        print('\nReloading the model...',end='')
        m = torch.load(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves',target)))['model']
        m.to(device)
        print('loaded :',target,'!')
    else:
        if autocomplete=='CONTINUE':
            idx = res
        elif autocomplete!='':
            idx = torch.tensor([tokenizer.eot_token]+tokenizer.encode(autocomplete), dtype=torch.long, device=device)
            idx = idx.reshape((1, len(idx)))
            prompt_end = len(idx[0])
        else:
            idx = torch.zeros((1,1), dtype=torch.long, device=device)
            idx[0,0]=tokenizer.eot_token
            prompt_end = len(idx[0])
        print('\033[32m'+tokenizer.decode(idx[0][:prompt_end]),end='')
        print('\033[0m'+tokenizer.decode(idx[0][prompt_end:]),end='\033[35m')
        res = idx
        for i in range(max_tokens):
            res = m.generate(idx=res, max_new_tokens=1)
            print(tokenizer.decode(res[0,i+len(idx[0]):]),end='')
            sys.stdout.flush()
        print('\033[0m\n')
    autocomplete = input("Type in some text to autocomplete (STOP to stop) : ")

