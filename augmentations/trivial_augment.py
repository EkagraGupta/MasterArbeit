import torch
from torchvision.transforms import TrivialAugmentWide
from PIL import Image
from typing import Optional

from utils.custom_trivial_augment import CTrivialAugmentWide
from utils import comparison_metrics
from augmentations.random_crop import RandomCrop


class CustomTrivialAugmentWide:
    """A class to apply custom or standard trivial augmentations to images. This class can
    also compute confidence scores based on the type of augmentation applied.

    Attributes:
        custom (bool): Flag to indicate if custom augmentation should be used.
    """

    def __init__(
        self, custom: bool = False, augmentation_name: str = None, severity: int = 0, get_signed: bool = False
    ):
        self.custom = custom
        self.augmentation_name = augmentation_name
        self.severity = severity
        self.get_signed = get_signed
        self.chance = 1 / 10  # cifar10
        # self.chance = 1 / 100  # cifar100

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

        trivial_augment = CTrivialAugmentWide(
            augmentation_name=self.augmentation_name, severity=self.severity, get_signed=self.get_signed
        )
        augment_im, augment_info = trivial_augment(im)
        augmentation_type = next(iter(augment_info.keys()))
        augmentation_magnitude = augment_info[augmentation_type]

        # SSIM Calculation
        # confidence_aa = comparison_metrics.structural_similarity_calculation(
        #     im, augment_im)

        # Structural SSIM calculation
        # confidence_aa = comparison_metrics.multiscale_structural_similarity(im, augment_im)

        # Normalized Cross Correlation calculation
        # confidence_aa = comparison_metrics.normalized_cross_correlation(im, augment_im)

        # Spatial Correlation Coefficient calculation
        # confidence_aa = comparison_metrics.spatial_correlation_coefficient(im, augment_im)

        # Universal Image Quality Index calculation
        # confidence_aa = comparison_metrics.universal_image_quality_index(im, augment_im)

        # Visual Information Fidelity calculation
        # confidence_aa = comparison_metrics.visual_information_fidelity(im, augment_im)

        if augmentation_type == 'ShearX':
            confidence_aa = comparison_metrics.gaussian(
                augmentation_magnitude, a=1.0, b=0.0, c=0.56)
        elif augmentation_type == 'ShearY':
            confidence_aa = comparison_metrics.gaussian(
                augmentation_magnitude, a=1.0, b=0.02, c=0.56)
        elif augmentation_type == 'TranslateX':
            dim1, dim2 = im.size[0], im.size[1]
            tx = augment_info[augmentation_type]
            # print(f'dim1: {dim1}, dim2: {dim2}, tx: {tx}')
            random_crop = RandomCrop()
            visibility = random_crop.compute_visibility(
                dim1=dim1, dim2=dim2, tx=tx, ty=0
            )
            k = 3
            confidence_aa = 1 - (1 - self.chance) * (1 - visibility) ** k
        elif augmentation_type == 'TranslateY':
            dim1, dim2 = im.size[0], im.size[1]
            ty = augment_info[augmentation_type]
            random_crop = RandomCrop()
            visibility = random_crop.compute_visibility(
                dim1=dim1, dim2=dim2, tx=0, ty=ty
            )
            k = 2
            confidence_aa = 1 - (1 - self.chance) * (1 - visibility) ** k
        elif augmentation_type == 'Brightness':
            # confidence_aa = comparison_metrics.custom_function(augmentation_magnitude, 1.2438093, 7.18937766, -0.87255438, -0.0573816, -0.2456411)
            confidence_aa = comparison_metrics.sigmoid(
                augmentation_magnitude, 0.9753, 17.0263, -0.8297)
        elif augmentation_type == 'Contrast':
            confidence_aa = comparison_metrics.sigmoid(
                augmentation_magnitude, 0.9914758, 13.89562814, -0.82550186)
        elif augmentation_type == 'Color':
            confidence_aa = comparison_metrics.sigmoid(
                augmentation_magnitude, 1.0, 4.93537641, -1.5837580)
        elif augmentation_type == 'Sharpness':
            confidence_aa = comparison_metrics.sigmoid(
                augmentation_magnitude, 0.9995181, 7.07685057, -1.24349678)
        elif augmentation_type == 'Posterize':
            confidence_aa = comparison_metrics.multiscale_structural_similarity(im, augment_im)
        elif augmentation_type == 'Solarize':
            confidence_aa = comparison_metrics.spatial_correlation_coefficient(im, augment_im)
        else:
            confidence_aa = 1.0

        # print(f"\nAugmentation info: {augment_info}\tconf: {confidence_aa}\n")
        return augment_im, list((augmentation_magnitude, confidence_aa))
