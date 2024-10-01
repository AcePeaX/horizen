import torch


import sys
import os

dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../lib")
)
# setting path
sys.path.append(dir_path)


target = ''
if target=='':
    target = input('What is the target model : ')

m_full = torch.load(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves',target)))
m = m_full['model']
print('Loaded model')
