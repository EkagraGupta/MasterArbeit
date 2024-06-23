import torch
from torch import Tensor

def get_softlabel_tensor(label: Tensor, correction_factor: float):
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    label_dict = {}
    label_tensor = torch.zeros(len(classes))
    for i in range(len(classes)):
        if i==label:
            label_tensor[i] = correction_factor
        else:
            def_value = (1. - correction_factor) / (len(classes) - 1)
            label_tensor[i] = def_value

        label_dict[classes[i]] = label_tensor[i]
        # label_tensor[i] = 0.0 if i!=label else 1.
    return label_dict, label_tensor
