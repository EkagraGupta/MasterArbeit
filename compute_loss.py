import torch
from torch.nn import functional as F
import torch.nn as nn


def soft_loss(pred, label, confidence):
    log_prob = F.log_softmax(pred, dim=1)
    n_class = pred.size(1)

    # Make soft one-hot target
    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    print(f"\nOnehot: {one_hot}\n")
    label = label.unsqueeze(1)
    confidence = confidence.unsqueeze(1)
    one_hot.scatter_(dim=1, index=label, src=confidence)
    print(f"\nSoftened: {one_hot}\nSum: {torch.sum(one_hot)}")

    # Compute weighted KL loss
    kl = confidence * F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
    return kl.mean()


if __name__ == "__main__":
    outputs = torch.tensor(
        [
            [
                -1.2265,
                -0.3036,
                0.5293,
                -0.7486,
                0.8621,
                0.3470,
                0.5692,
                -0.6229,
                -0.8170,
                1.3341,
            ]
        ]
    )
    labels = torch.tensor([0])
    confidence = torch.tensor([1.])

    s_loss = soft_loss(pred=outputs, label=labels, confidence=confidence)
    print(s_loss)

    criterion = nn.CrossEntropyLoss()
    h_loss = criterion(outputs, labels)
    print(h_loss)
