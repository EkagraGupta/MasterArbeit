import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as ff
from typing import Optional, List
import numpy as np

from dump.dataset import load_dataset


class SoftAugment:
    """A class to perform soft augmentation on images."""

    def __init__(
        self,
        aa_info: dict = {'None': 10},
        n_class: int = 10,
        k: int = 2,
        bg_crop: float = 0.01,
        sigma_crop: float = 10,
    ):
        """Initializes the SoftAugment class with specific parameters.

        Args:
            n_class (int, optional): Number of classes. Defaults to 10.
            k (int, optional): Non linear power factor. Defaults to 2.
            bg_crop (float, optional): Background crop value. Defaults to 0.01.
            sigma_crop (float, optional): standard deviation for Gaussian offset. Defaults to 10.
        """
        self.n_class = n_class
        self.chance = 1 / n_class
        self.k = k
        self.aa_info = aa_info

        # crop parameters
        self.sigma_crop = sigma_crop
        self.bg_crop = bg_crop

    def draw_offset(self, sigma=0.3, limit=24, n=100):
        """Draws an integer offset from a clipped Guassian Distribution.

        Args:
            sigma (float, optional): Standard deviation. Defaults to 0.3.
            limit (int, optional): Limit to clip the Gaussian. Defaults to 24.
            n (int, optional): Number of attempts. Defaults to 100.

        Returns:
            int: The drawn offset.
        """
        for _ in range(n):
            x = torch.randn((1)) * sigma
            if abs(x) <= limit:
                return int(x)
        return int(0)

    def compute_visibility(self, dim1, dim2, tx, ty):
        """Computes the visibility factor based on translation offsets.

        Args:
            dim1 (int): Dimension 1 of the image.
            dim2 (int): Dimension 2 of the image.
            tx (int): Translation offset in x-direction.
            ty (int): Translation offset in y-direction.

        Returns:
            float: The visibility factor.
        """
        return (dim1 - abs(tx)) * (dim2 - abs(ty)) / (dim1 * dim2)

    def __call__(self, image, aa_info):
        """Applies the soft augmentation to the input images.

        Args:
            image (torch.Tensor): The input image tensor
            aa_info (dict): Augmentation info from Aggressive augmentations

        Returns:
            tuple: The augmented image and the confidence score.
        """
        dim1, dim2 = image.size(1), image.size(2)

        # create background
        bg = torch.ones((3, dim1 * 3, dim2 * 3)) * self.bg_crop * torch.randn((3, 1, 1))
        bg[:, dim1 : dim1 * 2, dim2 : dim2 * 2] = image  # put image at the center

        tx, ty = self.draw_offset(self.sigma_crop, dim1), self.draw_offset(
            self.sigma_crop, dim2
        )

        left, right = tx + dim1, tx + dim1 * 2
        top, bottom = ty + dim2, ty + dim2 * 2

        visibility = self.compute_visibility(dim1, dim2, tx, ty)
        confidence = 1 - (1 - self.chance) * (1 - visibility) ** self.k         # The non-linear function
        print(f'Initial Confidence: {confidence}')
        confidence = np.clip(abs(confidence * next(iter(aa_info.values()))), 0, 1)
        print(f'After Aggressive: {confidence}')
        
        cropped_image = bg[:, left:right, top:bottom]

        return cropped_image, confidence


def soft_target(pred: torch.Tensor, label: torch.Tensor, confidence: float):
    """Generates soft target labels and computes the weighted KL divergence loss.

    Args:
        pred (torch.Tensor): The predicted logits.
        label (torch.Tensor): The true labels.
        confidence (float, optional): The confidence scores.

    Returns:
        torch.Tensor: The computed loss.
    """
    label = label.unsqueeze(1)
    confidence = torch.tensor(confidence).view(-1, 1)

    log_prob = F.log_softmax(pred, dim=1)
    n_class = pred.size(1)

    # make soft one-hot target
    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    confidence_expanded = confidence.expand_as(one_hot)
    one_hot.scatter_(dim=1, index=label, src=confidence_expanded)
    print(f"Soft Label: {one_hot}\n")

    # compute weighted KL loss
    kl = confidence * F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
    return kl.mean()


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    trainloader, _, classes = load_dataset(batch_size=1, transform=transform)

    from load_augmented_dataset import get_training_dataloader

    custom_trainloader = get_training_dataloader(num_samples=10, shuffle=True)

    images, labels, confidence = next(iter(custom_trainloader))
    print(f"\nOriginal Hard label: {labels} -> {classes[labels.item()]}\n")
    soft_augment = SoftAugment()

    new_image, confidence = soft_augment(images[0])
    pil_new_image = ff.to_pil_image(new_image)
    pil_new_image.save(
        "/home/ekagra/Desktop/Study/MA/code/example/example_augmented_image.png"
    )

    print(f"Confidence: {confidence}\n")

    outputs = torch.tensor(
        [
            [
                0.01,  # 0: airplane
                0.01,  # 1: automobile
                0.01,  # 2: bird
                0.01,  # 3: cat
                0.01,  # 4: deer
                0.01,  # 5: dog
                0.91,  # 6: frog
                0.01,  # 7: horse
                0.01,  # 8: ship
                0.01,  # 9: truck
            ],
        ]
    )

    loss_param = soft_target(pred=outputs, label=labels, confidence=confidence)

    print(f"Loss param: {loss_param}")
