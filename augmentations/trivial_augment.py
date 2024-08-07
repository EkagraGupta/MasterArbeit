import torch
from torchvision.transforms import TrivialAugmentWide
from PIL import Image
from typing import Optional

from utils.custom_trivial_augment import CTrivialAugmentWide
from utils.sift_comparison import sift_correction_factor
from utils.orb_comparison import orb_correction_factor
from utils.ssim_comparison import ssim_operation
from utils.ncc import normalized_cross_correlation
from utils.vif import compute_vif
from augmentations.random_crop import RandomCrop


class CustomTrivialAugmentWide:
    """A class to apply custom or standard trivial augmentations to images. This class can
    also compute confidence scores based on the type of augmentation applied.

    Attributes:
        custom (bool): Flag to indicate if custom augmentation should be used.
    """

    def __init__(
        self, custom: bool = False, augmentation_name: str = None, severity: int = 0
    ):
        self.custom = custom
        self.augmentation_name = augmentation_name
        self.severity = severity
        self.chance = 1 / 10  # number of classes

    def __call__(self, im: Optional[Image.Image]) -> Optional[tuple]:
        """Applies the augmentation to the given image.

        Args:
            im (Optional[Image.Image]): Input image to be augmented.

        Returns:
            Optional[tuple]: The augmented image and optionally the confidence scores.
        """
        if self.custom:
            augment_im, augment_info = self.get_augment_info(self, im)
            return augment_im, augment_info
        else:
            trivial_augmentation = TrivialAugmentWide()
            augment_im = trivial_augmentation(im)
            return augment_im

    @staticmethod
    def get_augment_info(self, im: torch.tensor) -> tuple:
        """Applies a custom trivial augmentation and computes a confidence score.

        Args:
            im (torch.tensor): Input image to be augmented.

        Returns:
            tuple: The augmented image and the computed confidence score.
        """
        pixelwise_augs = [
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

        trivial_augment = CTrivialAugmentWide(
            augmentation_name=self.augmentation_name, severity=self.severity
        )
        augment_im, augment_info = trivial_augment(im)
        augmentation_type = next(iter(augment_info.keys()))

        confidence_aa = ssim_operation(im1=im, im2=augment_im)
        if augmentation_type == "TranslateX":
            dim1, dim2 = im.size[0], im.size[1]
            tx = augment_info[augmentation_type]
            # print(f'dim1: {dim1}, dim2: {dim2}, tx: {tx}')
            random_crop = RandomCrop()
            visibility = random_crop.compute_visibility(
                dim1=dim1, dim2=dim2, tx=tx, ty=0
            )
            k = 3
            confidence_aa = 1 - (1 - self.chance) * (1 - visibility) ** k
        elif augmentation_type == "TranslateY":
            dim1, dim2 = im.size[0], im.size[1]
            ty = augment_info[augmentation_type]
            random_crop = RandomCrop()
            visibility = random_crop.compute_visibility(
                dim1=dim1, dim2=dim2, tx=0, ty=ty
            )
            k = 3
            confidence_aa = 1 - (1 - self.chance) * (1 - visibility) ** k
        elif augmentation_type == "ShearX":
            confidence_aa = sift_correction_factor(original_image=im, augmented_image=augment_im)
        # print(f"\nAugmentation info: {augment_info}\tconf: {confidence_aa}\n")
        return augment_im, confidence_aa
