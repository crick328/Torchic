import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.0")

from torchic import Tensor, nn
import numpy as np

def cross_entropy_loss_numpy(input, target, weight=None, reduction='mean', ignore_index=-100, label_smoothing=0.0):
    """
    Computes the Cross-Entropy loss between input tensor and target tensor.

    Parameters:
        input (ndarray): Input tensor (predictions).
                         Shape: (batch_size, num_classes)
        target (ndarray): Target tensor (ground truth labels).
                          Shape: (batch_size,)
        weight (ndarray, optional): A manual rescaling weight given to each class.
                                    It has the same dimensions as the number of classes.
                                    Default: None
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'.
                                   'none': no reduction will be applied,
                                   'mean': the sum of the output will be divided by the number of elements in the output,
                                   'sum': the output will be summed.
                                   Default: 'mean'
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient.
                                      Default: -100
        label_smoothing (float, optional): Applies label smoothing to the raw scores/targets.
                                           Default: 0.0

    Returns:
        ndarray: Computed Cross-Entropy loss.
    """
    # Apply label smoothing if specified
    num_classes = input.shape[1]
    smoothed_target = np.full_like(input, label_smoothing / (num_classes - 1))
    smoothed_target[np.arange(len(target)), target] = 1 - label_smoothing

    # Handle ignore_index
    mask = (target != ignore_index)

    # Compute log probabilities
    #log_prob = np.log(input + 1e-20)  # small epsilon to prevent log(0)
    log_soft = nn.LogSoftmax(-1)
    log_prob = log_soft(Tensor(input)).data
    cross_entropy = -np.sum(smoothed_target * log_prob, axis=-1)

    # Apply weights if given
    if weight is not None:
        cross_entropy *= weight[target]

    # Apply mask and reduction
    match(reduction):
        case 'none':
            loss = cross_entropy * mask
        case 'mean':
            loss = np.sum(cross_entropy * mask) / np.sum(mask)
            #loss = np.mean(cross_entropy * mask)
        case 'sum':
            loss = np.sum(cross_entropy * mask)
        case _:
            raise ValueError("Unsupported reduction mode. Please use 'none', 'mean', or 'sum'.")

    return loss


values = [
    [0.2, 0.3, 0.5],  # Logits per le tre classi del primo campione
    [0.1, 0.6, 0.3],  # Logits per le tre classi del secondo campione
    [0.4, 0.2, 0.4]   # Logits per le tre classi del terzo campione
]
#values = [0.2, 0.3, 0.5]
true_val = [2, 1, 1]

print(cross_entropy_loss_numpy(np.array(values), np.array(true_val), ignore_index=-100, reduction="none", label_smoothing=0.5))