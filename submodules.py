import importlib.util
import os
import sys


def add_path(subpath):
    libpath = os.path.abspath(subpath)
    if libpath not in sys.path:
        sys.path.append(libpath)

add_path('./pytorch-pretrained-bert')

from pytorch_pretrained_bert import optimization_openai

OpenAIAdam = optimization_openai.OpenAIAdam

