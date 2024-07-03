import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# setting path
sys.path.append(dir_path)

from base_model import BigramLanguageBaseModel
