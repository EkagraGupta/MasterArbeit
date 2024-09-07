import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np


def soft_loss2(pred, label, confidence):
    n_class = pred.size(1)
    print(
        f'label: {label.shape}\nconfidence: {confidence.shape}\npred: {pred.shape}')
    one_hot = torch.full_like(pred, fill_value=(
        confidence / (n_class - 1)).item())
    one_hot.scatter_(dim=1, index=label.unsqueeze(1),
                     src=confidence.unsqueeze(1))
    log_prob = F.log_softmax(pred, dim=1)
    print(f'one_hot: {one_hot}\nlog_prob: {log_prob.shape}')

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).mean()


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)
    print(f'one_hot: {one_hot}')
    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).mean()


def soft_loss_paper(pred, label, confidence):
    label = label.unsqueeze(1)
    target = label.long()
    prob = (1 - (label - target))
    n_class = pred.size(1)
    weight = torch.clone(prob).float()
    weight = confidence.unsqueeze(1)

    one_hot = torch.zeros_like(pred)
    one_hot.scatter_(dim=1, index=torch.ones_like(target) *
                     (n_class-1), src=(1 - prob.float()) * 1.0)
    one_hot.scatter_(dim=1, index=target, src=prob.float())
    log_prob = F.log_softmax(pred, dim=1)
    kl = weight * F.kl_div(input=log_prob, target=one_hot,
                           reduction='none').sum(-1)
    return kl.mean()


def ECE(y_true, y_pred, num_bins=11):
    y_pred = torch.nn.functional.softmax(y_pred)
    # print(y_pred.shape)
    y_p = np.squeeze(y_pred.cpu().numpy())
    y_t = np.squeeze(y_true.cpu().numpy())
    pred_y = np.argmax(y_p, axis=-1)
    # print(pred_y)
    correct = (pred_y == y_t).astype(np.float32)
    prob_y = np.max(y_p, axis=-1)
    # print(prob_y.shape)
    # print(y_t.shape)
    bins = np.linspace(start=0, stop=1.0, num=num_bins)
    binned = np.digitize(prob_y, bins=bins, right=True)

    errors = np.zeros(num_bins)
    confs = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    corrects = np.zeros(num_bins)
    accs = np.zeros(num_bins)
    o = 0
    for b in range(num_bins):
        mask = binned == b
        # if np.any(mask):
        count = np.sum(mask)
        counts[b] = count
        corrects[b] = np.sum(correct[mask])
        if count > 0:
            accs[b] = corrects[b] / counts[b]
            confs[b] = np.mean(prob_y[mask])
            errors[b] = np.abs(accs[b] - np.mean(prob_y[mask])) * counts[b]

    return np.sum(errors)/y_pred.shape[0], confs, accs, np.array(counts)


def soft_loss(pred, label, confidence):
    log_prob = F.log_softmax(pred, dim=1)
    print(f'log_prob: {log_prob.shape}')
    n_class = pred.size(1)

    # Make soft one-hot target
    label = label.unsqueeze(1)
    print(f'label: {label.shape}')
    confidence = confidence.unsqueeze(1).float()
    print(f'confidence: {confidence}')
    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    print(f'one_hot: {one_hot.shape}')
    one_hot.scatter_(dim=1, index=label, src=confidence)
    print(f'one_hot: {one_hot}')

    # Compute weighted KL loss
    kl_uw = F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
    kl_uw = kl_uw.unsqueeze(1)
    print(f'kl_unweighted: {kl_uw.shape}')                   # Unweighted
    kl = confidence * kl_uw  # Weighted
    print(f'kl: {kl.shape}')
    return kl.mean()


if __name__ == "__main__":
    # Test the soft loss function
    labels = torch.tensor([2, 0, 3, 7, 7])
    outputs = torch.tensor([
        [0.1730,  0.1881,  0.0669,  0.2474,  0.1378,  0.5475,  0.1066,  0.3351,
         0.0911, -0.0094],
        [0.1765,  0.1779,  0.1047,  0.3240,  0.1645,  0.4601,  0.0828,  0.3835,
         0.1566, -0.0390],
        [0.1442,  0.1395,  0.0956,  0.2689,  0.0836,  0.3936,  0.0026,  0.1914,
         0.0504,  0.0041],
        [0.1363,  0.1181, -0.0057,  0.3126,  0.1279,  0.4747,  0.0967,  0.2872,
         0.0372,  0.0938],
        [0.1836,  0.1860,  0.1349,  0.3351,  0.0847,  0.4231,  0.0219,  0.3322,
         0.0351,  0.0358]
    ])
    confidences = torch.tensor([0.9994, 0.9919, 1.0000, 1.0000, 0.9804])

    # Compute the soft loss
    loss = soft_loss(outputs, labels, confidences)
    print(f"Soft loss: {loss.item()}")

    # Cross-entropy loss
    loss2 = F.cross_entropy(outputs, labels)
    print(f"Cross-entropy loss: {loss2.item()}")
