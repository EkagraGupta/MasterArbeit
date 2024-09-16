import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np

def soft_loss(pred, label, confidence, reweight=False):
    log_prob = F.log_softmax(pred, dim=1)
    n_class = pred.size(1)

    # Make soft one-hot target
    label = label.unsqueeze(1)
    confidence = confidence.unsqueeze(1).float()
    # soft one_hot
    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    one_hot.scatter_(dim=1, index=label, src=confidence)
    # print(f'soft one_hot: {one_hot}')
    # hard one_hot
    # one_hot = torch.zeros_like(pred)
    # one_hot.scatter_(dim=1, index=label, value=1.0)
    # Compute weighted KL loss
    kl = F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
    kl = kl.unsqueeze(1)  # Unweighted
    if reweight:
        kl = confidence * kl  # Weighted
    return kl.mean()

# def cross_entropy_loss(pred, label):
#     log_prob = F.log_softmax(pred, dim=1)
#     n_class = pred.size(1)

#     label = label.unsqueeze(1)
#     one_hot = torch.zeros_like(pred).scatter_(dim=1, index=label, value=1.0)
#     kl_uw = F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
#     kl_uw = kl_uw.unsqueeze(1)

#     return kl_uw.mean()

# import numpy as np

def cross_entropy_loss(y_pred, label, confidence=None):
    n_class = y_pred.size(1)

    # Convert the label to one-hot encoding
    if confidence is not None:
        confidence = confidence.unsqueeze(1).float()
        label = label.unsqueeze(1)
        y_true = torch.ones_like(y_pred) * (1 - confidence) / (n_class - 1)
        y_true.scatter_(dim=1, index=label, src=confidence)
    else:
        y_true = torch.zeros(label.size(0), n_class)
        y_true.scatter_(1, label.unsqueeze(1), 1)
    
    # If y_pred are logits, apply softmax to get probabilities
    y_pred = F.softmax(y_pred, dim=1)
    
    # Clipping predictions to avoid log(0)
    y_pred = torch.clamp(y_pred, 1e-12, 1. - 1e-12)
    
    # Calculating the cross-entropy loss
    loss = -torch.sum(y_true * torch.log(y_pred)) / y_true.size(0)
    
    return loss


if __name__ == "__main__":
    # Test the soft loss function
    labels = torch.tensor([2, 0, 3, 7, 7])
    outputs = torch.tensor(
        [
            [
                0.1730,
                0.1881,
                0.0669,
                0.2474,
                0.1378,
                0.5475,
                0.1066,
                0.3351,
                0.0911,
                -0.0094,
            ],
            [
                0.1765,
                0.1779,
                0.1047,
                0.3240,
                0.1645,
                0.4601,
                0.0828,
                0.3835,
                0.1566,
                -0.0390,
            ],
            [
                0.1442,
                0.1395,
                0.0956,
                0.2689,
                0.0836,
                0.3936,
                0.0026,
                0.1914,
                0.0504,
                0.0041,
            ],
            [
                0.1363,
                0.1181,
                -0.0057,
                0.3126,
                0.1279,
                0.4747,
                0.0967,
                0.2872,
                0.0372,
                0.0938,
            ],
            [
                0.1836,
                0.1860,
                0.1349,
                0.3351,
                0.0847,
                0.4231,
                0.0219,
                0.3322,
                0.0351,
                0.0358,
            ],
        ]
    )
    # confidences = torch.tensor([0.9994, 0.9919, 0.1, 1.0, 0.9804])
    confidences = torch.ones(outputs.size(0), dtype=torch.float32)

    # Compute the soft loss
    loss = soft_loss(outputs, labels, confidences, reweight=True)
    # loss = cross_entropy_loss(outputs, labels)
    print(f"Soft loss: {loss.item()}")

    # Cross-entropy loss
    loss2 = F.cross_entropy(outputs, labels)
    # loss2 = cross_entropy_loss(outputs, labels, confidence=confidences)
    print(f"Cross-entropy loss: {loss2.item()}")

    print(f'Soft loss and CE are equal?\t{bool(loss == loss2)}')