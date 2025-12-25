import torch
import numpy as np
import random
import os

def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_accuracy(logits, labels):
    """
    logits: (batch_size, 4)
    labels: (batch_size)
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    return correct