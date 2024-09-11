import torch
from torchvision import transforms
from typing import Optional
from PIL import Image


class RandomCrop:
    """A class to apply a random crop transformation to an image. This class
    is intended for use with images and can also compute confidence scores for
    the transformed image.

    Attributes:
    n_class (int): Number of classes for the classification task.
    k (int): Non-linear function parameter for confidence calculation.
    bg_crop (float): Background cropping intensity.
    sigma_crop (float): Standard deviation for drawing the offset.
    """

    def __init__(
        self,
        n_class: int = 10,
        k: int = 2,
        bg_crop: float = 0.01,
        sigma_crop: float = 10,
        dataset_name: str = "CIFAR10",
        custom: bool = False
    ):
        if dataset_name == "CIFAR10":
            self.n_class = 10
        elif dataset_name == "CIFAR100":
            self.n_class = 100
        else:
            raise ValueError(f"Dataset name {dataset_name} not supported")
        self.chance = 1 / self.n_class
        self.k = k
        self.sigma_crop = sigma_crop
        self.bg_crop = bg_crop
        self.custom = custom

    def draw_offset(
        self,
        sigma: Optional[float] = 0.3,
        limit: Optional[int] = 24,
        n: Optional[int] = 100,
    ) -> int:
        """Draws a random offset within a specified limit using a normal distribution.

        Args:
            sigma (float, optional): Standard deviation for the normal distribution. Defaults to 0.3.
            limit (int, optional): Maximum absolute value for the offset. Defaults to 24.
            n (int, optional): Number of attempts to draw a valid offset. Defaults to 100.

        Returns:
            int: The drawn offset within the limit
        """
        for _ in range(n):
            x = torch.randn((1)) * sigma
            if abs(x) <= limit:
                return int(x)
        return 0

    def compute_visibility(self, dim1: int, dim2: int, tx: int, ty: int) -> float:
        """Computes the visibility of the cropped uimage within the background.

        Args:
            dim1 (int): Height of the image.
            dim2 (int): Width of the image.
            tx (int): Horizontal offset.
            ty (int): Vertical offset.

        Returns:
            float: Visibility ratio of the cropped image.
        """
        return (dim1 - abs(tx)) * (dim2 - abs(ty)) / (dim1 * dim2)

    def __call__(self, image: Optional[Image.Image]) -> Optional[tuple]:
        """Applies the random crop transformation to the given image.

        Args:
            image (Optional[Image.Image]): Input image or a tuple containing the image
            and an additional confidence value.

        Returns:
            Optional[tuple]: The cropped image and the computed confidence values.
        """
        confidence_aa = 1.0

        if isinstance(image, tuple) and len(image) == 2 and isinstance(image[1], float):
            confidence_aa = image[1]
            image = image[0]
        elif (
            isinstance(image, tuple) and len(image) == 2 and isinstance(image[1], list)
        ):
            confidence_aa = image[1][1]
            image = image[0]

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        dim1, dim2 = image.size(1), image.size(2)

        # Create background
        bg = torch.zeros((3, dim1 * 3, dim2 * 3)) * self.bg_crop * torch.randn((3, 1, 1))
        bg[:, dim1 : dim1 * 2, dim2 : dim2 * 2] = image  # Put image at the center

        # calculate random offsets.
        tx, ty = self.draw_offset(self.sigma_crop, dim1), self.draw_offset(
            self.sigma_crop, dim2
        )

        # define the cropping boundaries.
        left, right = tx + dim1, tx + dim1 * 2
        top, bottom = ty + dim2, ty + dim2 * 2

        # crop the image
        cropped_image = bg[:, left:right, top:bottom]


        if self.custom:
            # compute visibility and confidence score
            visibility = self.compute_visibility(dim1, dim2, tx, ty)
            confidence_rc = (
                1 - (1 - self.chance) * (1 - visibility) ** self.k
            )  # The non-linear function
        else:
            confidence_rc = 1.0

        # combine confidence scores if available
        if confidence_aa is not None:
            # confidence_aa = torch.tensor(confidence_aa, dtype=torch.float32)
            confidences = (confidence_aa, confidence_rc)
        else:
            confidences = confidence_rc

        to_pil = transforms.ToPILImage()
        cropped_image = to_pil(cropped_image)
        # print(f'confidence_rc: {confidence_rc}\tconfidence_aa: {confidence_aa}\tconfidences: {confidences}')
        return cropped_image, confidences
