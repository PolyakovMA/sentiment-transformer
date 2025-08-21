import os
import random
import numpy as np
import torch
from transformers import set_seed
from sklearn.utils.class_weight import compute_class_weight

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


def get_class_weights(labels):

    labels_np = np.array(labels)
    unique_classes = np.unique(labels_np)

    class_weights_np = compute_class_weight(class_weight="balanced", classes=unique_classes, y=labels_np)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    return class_weights


