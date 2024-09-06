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
    one_hot.scatter_(dim=1, index=label, src=confidence)

    # Compute weighted KL loss
    kl = confidence * F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(
        -1
    )  # Weighted
    # kl = F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)                   # Unweighted
    
    return kl.mean()

def soft_loss2(pred, label, confidence):
    n_class = pred.size(1)
    print(f'label: {label.shape}\nconfidence: {confidence.shape}\npred: {pred.shape}')
    one_hot = torch.full_like(pred, fill_value=(confidence / (n_class - 1)).item())
    one_hot.scatter_(dim=1, index=label.unsqueeze(1), src=confidence.unsqueeze(1))
    log_prob = F.log_softmax(pred, dim=1)
    print(f'one_hot: {one_hot}\nlog_prob: {log_prob.shape}')

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).mean()

def smoothed_cross_entropy_loss(pred, label, confidence):
    """
    Computes the smoothed cross-entropy loss where confidence is used as the smoothing factor.

    Args:
        pred (torch.Tensor): Predictions of the model (logits) of shape (batch_size, num_classes).
        label (torch.Tensor): Ground truth labels of shape (batch_size,).
        confidence (torch.Tensor): Confidence scores for each image of shape (batch_size,).
        
    Returns:
        torch.Tensor: The smoothed cross-entropy loss.
    """
    
    # Number of classes
    num_classes = pred.size(1)
    
    # Apply label smoothing using confidence as the smoothing factor
    one_hot_labels = F.one_hot(label, num_classes).float()
    smoothing = 1 - confidence.unsqueeze(1)
    smooth_labels = one_hot_labels * confidence.unsqueeze(1) + smoothing / num_classes
    
    # Cross-entropy loss with smoothed labels
    log_probs = F.log_softmax(pred, dim=1)
    loss = -torch.sum(smooth_labels * log_probs, dim=1)
    
    # Return the mean loss
    return loss.mean()

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)
    print(f'one_hot: {one_hot}')
    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).mean()

if __name__ == "__main__":
    # Test the soft loss function
    pred = torch.randn(1, 10)
    label = torch.tensor([3])
    confidence = torch.tensor([0.5])
    s_loss = smooth_crossentropy(pred=pred, gold=label, smoothing=0.1)
    # s_loss = smoothed_cross_entropy_loss(pred, label, confidence)
    # s_loss2 = soft_loss2(pred, label, confidence)

    # Taking original loss as well
    criterion = nn.CrossEntropyLoss()
    h_loss = criterion(pred, label)

    print(f"\nSoft Loss: {s_loss:.3f}\nCross Entropy Loss: {h_loss:.3f}")
