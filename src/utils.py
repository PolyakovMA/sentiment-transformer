import os
import random
import numpy as np
import torch
from transformers import set_seed

def set_global_seed(seed: int = 42):
    """
    Фиксирует все генераторы случайных чисел (Python, NumPy, Torch, HF).
    """
    random.seed(seed)                  
    np.random.seed(seed)               
    torch.manual_seed(seed)            # PyTorch CPU
    torch.cuda.manual_seed_all(seed)   # PyTorch GPU
    set_seed(seed)                     

def save_model(trainer, path):
    os.makedirs(path, exist_ok=True)
    trainer.save_model(path)