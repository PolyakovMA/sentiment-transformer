import os
import random
import numpy as np
import torch
from transformers import set_seed
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss


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
    
    class_weights_np = compute_class_weight(class_weight="balanced", 
                                            classes=np.unique(labels), 
                                            y=labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    return class_weights


def custom_compute_loss(outputs, labels, class_weights):

    logits = outputs.logits
    loss_fct = CrossEntropyLoss(weight=class_weights.to(logits.device))
    
    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))