import torch
import torch.nn.functional as F


def soft_target(pred, label, confidence):
    label = label.unsqueeze(1)
    confidence = torch.tensor(confidence).view(-1, 1)

    log_prob = F.log_softmax(pred, dim=1)
    n_class = pred.size(1)

    # Make soft one-hot target
    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    one_hot.scatter_(dim=1, index=label, src=confidence)
    print(f"Softened: {one_hot}\n")

    # Compute weighted KL loss
    kl = confidence * F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
    return kl.mean()
