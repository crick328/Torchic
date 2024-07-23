import torch
from torch import Tensor, nn
import numpy as np

import numpy as np

def cross_entropy_loss(input: np.ndarray, target, label_smoothing=0.0, ignore_index=None, reduction='mean'):
    """
    Compute the cross-entropy loss with label smoothing and ignore index support.
    
    Args:
    - input (np.ndarray): Logits of shape (N, C) or (C,) where N is the number of samples
                          and C is the number of classes.
    - target (np.ndarray): Target labels of shape (N,) or a single value (int) for (C,) input.
    - label_smoothing (float): Amount of label smoothing.
    - ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
    - reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    
    Returns:
    - loss (float or np.ndarray): The calculated loss.
    """
    
    if input.ndim == 1:
        input = input.reshape(1, -1)
        target = np.array([target])
    
    N, C = input.shape

    # Label smoothing
    if label_smoothing > 0:
        smooth_target = (1 - label_smoothing) * np.eye(C)[target] + label_smoothing / C
    else:
        smooth_target = np.eye(C)[target]
    
    # Apply ignore_index
    if ignore_index is not None:
        mask = target != ignore_index
        input = input[mask]
        smooth_target = smooth_target[mask]
        target = target[mask]
    
    # Compute log probabilities
    log_probs = input - np.log(np.sum(np.exp(input), axis=-1, keepdims=True))
    
    # Compute the cross-entropy loss
    loss = -np.sum(smooth_target * log_probs, axis=1)
    
    # Apply reduction
    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    
    return loss

# Example usage:
input = np.array([
    [0.2, 0.3, 0.5],  # Logits per le tre classi del primo campione
    [0.1, 0.6, 0.3],  # Logits per le tre classi del secondo campione
    [0.4, 0.2, 0.4]   # Logits per le tre classi del terzo campione
])
input = np.array([0.2, 0.3, 0.5])
target = np.array([2, 1, 1])
reduction = "none"
ignore_index = 1
smooth=0.0
print(cross_entropy_loss(input, target, label_smoothing=smooth, ignore_index=ignore_index, reduction=reduction))
