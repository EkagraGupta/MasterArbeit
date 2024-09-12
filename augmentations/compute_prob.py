import torch
import numpy as np

def ComputeProb(x, T=0.8, n_classes=10, max_prob=1.0, pow=2.0):
    max_prob = torch.clamp_min(torch.tensor(max_prob),1/n_classes)
    if T <=0:
        T = 1e-10

    if x > T:
        return max_prob
    elif x > 0:
        a = (max_prob - 1/float(n_classes))/(T**pow)
        return max_prob - a * (T-x) ** pow
    else:
        return np.ones_like(x) * 1/n_classes