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


class CustomTrivialAugmentWide:
    """A class to apply custom or standard trivial augmentations to images. This class can
    also compute confidence scores based on the type of augmentation applied.

    Attributes:
        custom (bool): Flag to indicate if custom augmentation should be used.
    """

    def __init__(self, custom: bool = False):
        self.custom = custom

    def __call__(self, im: Optional[Image.Image]) -> Optional[tuple]:
        """Applies the augmentation to the given image.

        Args:
            im (Optional[Image.Image]): Input image to be augmented.

        Returns:
            Optional[tuple]: The augmented image and optionally the confidence scores.
        """
        if self.custom:
            augment_im, augment_info = self.get_augment_info(im)
            return augment_im, augment_info
        else:
            trivial_augmentation = TrivialAugmentWide()
            augment_im = trivial_augmentation(im)
            return augment_im

    @staticmethod
    def get_augment_info(im: torch.tensor) -> tuple:
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

        trivial_augment = CTrivialAugmentWide()
        augment_im, im_info = trivial_augment(im)
        # augmentation_type = next(iter(im_info.keys()))

        confidence_aa = ssim_operation(im1=im, im2=augment_im)
        # if augmentation_type in pixelwise_augs:
        #     # calculate VIF for pixel-wise augmentations
        #     # vif_value = compute_vif(im1=im, im2=augment_im)
        #     # confidence_aa = vif_value.item()
        #     # confidence_aa = 0.5
        # else:
        # #     # calculate SIFT correction factor for geometric transformations
        #     # confidence_aa = sift_correction_factor(original_image=im, augmented_image=augment_im)
        # #     # confidence_aa = orb_correction_factor(original_image=im, augmented_image=augment_im)
        # #     confidence_aa = ssim_operation(im1=im, im2=augment_im)
        #     # confidence_aa = normalized_cross_correlation(im1=im, im2=augment_im)
        #     # confidence_aa = 0.5
        # print(f"\nAugmentation info: {im_info}\tconf: {confidence_aa}\n")
        return augment_im, confidence_aa
