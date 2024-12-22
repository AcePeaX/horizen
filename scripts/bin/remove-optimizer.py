import torch

import sys
import os

dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../lib")
)
# setting path
sys.path.append(dir_path)


from models import BasicSelfAttentionLanguageModel



if len(sys.argv)<=2:
    print("Should specify 2 arguments, got only",(len(sys.argv)-1))
    exit()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

m = BasicSelfAttentionLanguageModel(10,10,10,context_size=10)

epoch, _ = m.load(sys.argv[1], map_location=device)


m.save(sys.argv[2], epoch=epoch)