import torch
from torch.nn import functional as F
import torch.nn as nn


def soft_loss(pred, label, confidence):
    log_prob = F.log_softmax(pred, dim=1)
    n_class = pred.size(1)

    # Make soft one-hot target
    label = label.unsqueeze(1)
    confidence = confidence.unsqueeze(1).float()
    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    # print(f"\nOnehot: {one_hot}\n")
    one_hot.scatter_(dim=1, index=label, src=confidence)
    # print(f"\nSoftened: {one_hot}\nSum: {torch.sum(one_hot)}")

    # Compute weighted KL loss
    # kl = confidence * F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)

    kl = confidence * F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
    return kl.mean()


# def soft_loss(pred, label, confidence):
#     label = label.unsqueeze(1)
#     target = label.long()
#     confidence = confidence.unsqueeze(1).float()
#     prob = (1 - (label - target))
#     weight = torch.ones_like(prob).float()
#     n_classes = pred.size(1)

#     one_hot = (torch.ones_like(pred) * (1 - confidence) / (n_classes - 1)).float()
#     one_hot.scatter_(dim=1, index=target, src=confidence)
#     log_prob = F.log_softmax(pred, dim=1)
#     kl = F.kl_div(input=log_prob.float(), target=one_hot.float(), reduction='none').sum(-1)
#     return kl.mean()


if __name__ == "__main__":
    # Test the soft loss function
    pred = torch.randn(1, 10)
    label = torch.tensor([3])
    confidence = torch.tensor([0.9])
    s_loss = soft_loss(pred, label, confidence)

    # Taking original loss as well
    criterion = nn.CrossEntropyLoss()
    h_loss = criterion(pred, label)

    print(f'\nSoft Loss: {s_loss:.3f}\nCross Entropy Loss: {h_loss:.3f}')
