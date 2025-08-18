import numpy as np
from sklearn.metrics import f1_score
import torch

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)

    preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    f1 = f1_score(labels, preds, average='weighted')
    return {"f1": f1}
