import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as ff
import numpy as np

from dump.dataset import load_dataset


class SoftAugment:
    """A class to perform soft augmentation on images."""

    def __init__(
        self,
        aa_info: dict = {"None": 10},
        n_class: int = 10,
        k: int = 2,
        bg_crop: float = 0.01,
        sigma_crop: float = 10,
    ):
        """
        Initializes the SoftAugment class with specific parameters.

        Args:
            aa_info (dict, optional): Augmentation info. Defaults to {"None": 10}.
            n_class (int, optional): Number of classes. Defaults to 10.
            k (int, optional): Non-linear power factor. Defaults to 2.
            bg_crop (float, optional): Background crop value. Defaults to 0.01.
            sigma_crop (float, optional): Standard deviation for Gaussian offset. Defaults to 10.
        """
        self.n_class = n_class
        self.chance = 1 / n_class
        self.k = k
        self.aa_info = aa_info
        self.pixelwise_augs = [
            "Invert",
            "Equalize",
            "AutoContrast",
            "Posterize",
            "Solarize",
            "SolarizeAdd",
            "Color",
            "Contrast",
            "Brightness",
            "Sharpness",
        ]
        self.sigma_crop = sigma_crop
        self.bg_crop = bg_crop

    def draw_offset(self, sigma=0.3, limit=24, n=100):
        """
        Draws an integer offset from a clipped Gaussian Distribution.

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
        return 0

    def compute_visibility(self, dim1, dim2, tx, ty):
        """
        Computes the visibility factor based on translation offsets.

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
        """
        Applies the soft augmentation to the input images.

        Args:
            image (torch.Tensor): The input image tensor.
            aa_info (dict): Augmentation info from aggressive augmentations.

        Returns:
            tuple: The augmented image and the confidence score.
        """
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
        confidence = (
            1 - (1 - self.chance) * (1 - visibility) ** self.k
        )  # The non-linear function

        augmentation_type = next(iter(aa_info.keys()))
        print(f"confidence: {confidence}\taa: {aa_info[augmentation_type]}\n")
        if augmentation_type in self.pixelwise_augs:
            confidence = np.clip(
                abs(np.mean([aa_info[augmentation_type], confidence])), 0, 1
            )
        else:
            confidence = np.clip(
                abs(np.mean([aa_info[augmentation_type], confidence])), 0, 1
            )
        return cropped_image, confidence


def soft_target(pred: torch.Tensor, label: torch.Tensor, confidence: float):
    """
    Generates soft target labels and computes the weighted KL divergence loss.

    Args:
        pred (torch.Tensor): The predicted logits.
        label (torch.Tensor): The true labels.
        confidence (float): The confidence scores.

    Returns:
        torch.Tensor: The computed loss.
    """
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


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    trainloader, _, classes = load_dataset(batch_size=1, transform=transform)

    from load_augmented_dataset import get_dataloader

    custom_trainloader = get_dataloader(num_samples=10, shuffle=False)

    images, labels, _ = next(iter(custom_trainloader))
    print(f"Original Hard label: {labels} -> {classes[labels.item()]}")

    from trivial_augment import CustomTrivialAugmentWide

    aa_transform = CustomTrivialAugmentWide()
    image, aa_info = aa_transform(images[0])
    soft_augment = SoftAugment()

    new_image, confidence = soft_augment(images[0], aa_info=aa_info)
    pil_new_image = ff.to_pil_image(new_image)
    pil_new_image.save(
        "/home/ekagra/Desktop/Study/MA/code/example/example_augmented_image.png"
    )

    outputs = torch.tensor(
        [[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.91, 0.01, 0.01, 0.01]]
    )
    loss_param = soft_target(pred=outputs, label=labels, confidence=confidence)

    print(f"Loss param: {loss_param}")
