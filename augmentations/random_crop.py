import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as ff
import numpy as np

from utils.dataset import load_dataset


class RandomCrop:

    def __init__(
        self,
        n_class: int = 10,
        k: int = 2,
        bg_crop: float = 0.01,
        sigma_crop: float = 10,
    ):
        self.n_class = n_class
        self.chance = 1 / n_class
        self.k = k
        self.sigma_crop = sigma_crop
        self.bg_crop = bg_crop

    def draw_offset(self, sigma=0.3, limit=24, n=100):
        for _ in range(n):
            x = torch.randn((1)) * sigma
            if abs(x) <= limit:
                return int(x)
        return 0

    def compute_visibility(self, dim1, dim2, tx, ty):
        return (dim1 - abs(tx)) * (dim2 - abs(ty)) / (dim1 * dim2)

    def __call__(self, image):
        confidence_aa = None
        if isinstance(image, tuple) and len(image) == 2 and isinstance(image[1], float):
            confidence_aa = image[1]
            image = image[0]

        dim1, dim2 = image.size(1), image.size(2)

        # Create background
        bg = torch.ones((3, dim1 * 3, dim2 * 3)) * self.bg_crop * torch.randn((3, 1, 1))
        bg[:, dim1 : dim1 * 2, dim2 : dim2 * 2] = image  # Put image at the center

        tx, ty = self.draw_offset(self.sigma_crop, dim1), self.draw_offset(
            self.sigma_crop, dim2
        )

        left, right = tx + dim1, tx + dim1 * 2
        top, bottom = ty + dim2, ty + dim2 * 2

        cropped_image = bg[:, left:right, top:bottom]

        visibility = self.compute_visibility(dim1, dim2, tx, ty)
        confidence_rc = (
            1 - (1 - self.chance) * (1 - visibility) ** self.k
        )  # The non-linear function

        if confidence_aa is not None:
            confidence_aa = torch.tensor(confidence_aa, dtype=torch.float32)
            confidences = (confidence_aa, confidence_rc)
        else:
            confidences = (confidence_rc,)
        print(confidence_rc)
        return cropped_image, confidences


def soft_target(pred: torch.Tensor, label: torch.Tensor, confidence: float):
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
