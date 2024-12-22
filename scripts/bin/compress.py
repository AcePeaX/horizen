import torch
import zipfile
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# setting path
sys.path.append(os.path.join(dir_path, "../lib"))

import models

target_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../saves'))


target = input('What is the target model : ')
output = input('What is the output file name : ')

conserveOptim = input('Do you want to conserve the optimizing data (yes or 1) : ')
if conserveOptim=='yes' or conserveOptim=='1':
    conserveOptim=True
else:
    conserveOptim=False




toWrite = os.path.join(target_path, target)
if not conserveOptim:
    m_full = torch.load(os.path.join(target_path, target))
    toWrite = toWrite + ".zipfiletemp"
    torch.save(m_full['model'],toWrite)


zip = zipfile.ZipFile(os.path.join(target_path, output), "w", zipfile.ZIP_DEFLATED)

zip.write(toWrite)
zip.close()
if not conserveOptim:
    os.remove(os.path.abspath(toWrite))