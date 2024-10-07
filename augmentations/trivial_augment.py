import torch
from PIL import Image
from typing import Optional
import numpy as np

from utils import comparison_metrics
from augmentations.random_crop import RandomCrop

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torchvision.transforms import functional as F, InterpolationMode
from model_confidence_mapping import model_accuracy_mapping
from hvs_augmentations import get_data


def _apply_op(
    im: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        im = F.affine(
            im,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        im = F.affine(
            im,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        im = F.affine(
            im,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        im = F.affine(
            im,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        im = F.rotate(im, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        im = F.adjust_brightness(im, 1.0 + magnitude)
    elif op_name == "Color":
        im = F.adjust_saturation(im, 1.0 + magnitude)
    elif op_name == "Contrast":
        im = F.adjust_contrast(im, 1.0 + magnitude)
    elif op_name == "Sharpness":
        im = F.adjust_sharpness(im, 1.0 + magnitude)
    elif op_name == "Posterize":
        im = F.posterize(im, int(magnitude))
    elif op_name == "Solarize":
        im = F.solarize(im, magnitude)
    elif op_name == "AutoContrast":
        im = F.autocontrast(im)
    elif op_name == "Equalize":
        im = F.equalize(im)
    elif op_name == "Invert":
        im = F.invert(im)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return im, {op_name: magnitude}


class CustomTrivialAugmentWide(torch.nn.Module):
    def __init__(
        self,
        custom: bool = False,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        severity: int = 0,
        augmentation_name: str = None,
        get_signed: bool = False,
        dataset_name: str = "CIFAR10",
    ):
        super().__init__()
        self.custom = custom
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

        """MODIFICATION: Add severity"""
        # self.severity = severity
        # self.augmentation_name = augmentation_name
        # self.get_signed = get_signed
        self.k = 2
        """MODIFICATION: Add severity"""

        if dataset_name == "CIFAR10":
            self.chance = 1 / 10
        elif dataset_name == "CIFAR100":
            self.chance = 1 / 100
        else:
            raise ValueError(f"Dataset name {dataset_name} not supported")

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

        # print(f'augmentation_name: {self.augmentation_name}\tseverity: {self.severity}')
        # if self.augmentation_name == "Identity":
        #     return {"Identity": (torch.tensor(0.0), False)}
        # elif self.augmentation_name == "ShearX":
        #     return {"ShearX": (torch.linspace(0.0, 0.99, num_bins), True)}
        # elif self.augmentation_name == "ShearY":
        #     return {"ShearY": (torch.linspace(0.0, 0.99, num_bins), True)}
        # elif self.augmentation_name == "TranslateX":
        #     return {"TranslateX": (torch.linspace(0.0, 32.0, num_bins), True)}
        # elif self.augmentation_name == "TranslateY":
        #     return {"TranslateY": (torch.linspace(0.0, 32.0, num_bins), True)}
        # elif self.augmentation_name == "Rotate":
        #     return {"Rotate": (torch.linspace(0.0, 135.0, num_bins), True)}
        # elif self.augmentation_name == "Brightness":
        #     return {"Brightness": (torch.linspace(0.0, 0.99, num_bins), True)}
        # elif self.augmentation_name == "Color":
        #     return {"Color": (torch.linspace(0.0, 0.99, num_bins), True)}
        # elif self.augmentation_name == "Contrast":
        #     return {"Contrast": (torch.linspace(0.0, 0.99, num_bins), True)}
        # elif self.augmentation_name == "Sharpness":
        #     return {"Sharpness": (torch.linspace(0.0, 0.99, num_bins), True)}
        # elif self.augmentation_name == "Posterize":
        #     return {
        #         "Posterize": (
        #             8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
        #             False,
        #         )
        #     }
        # elif self.augmentation_name == "Solarize":
        #     return {"Solarize": (torch.linspace(255.0, 0.0, num_bins), False)}
        # elif self.augmentation_name == "AutoContrast":
        #     return {"AutoContrast": (torch.tensor(0.0), False)}
        # elif self.augmentation_name == "Equalize":
        #     return {"Equalize": (torch.tensor(0.0), False)}
        # else:
        #     raise ValueError(
        #         f"The provided operator {self.augmentation_name} is not recognized."
        #     )

    def forward(self, im: torch.Tensor) -> Tensor:
        # if self.custom:
        augment_im, augment_info = self.apply_custom_augmentation(im)
        return augment_im, augment_info
        # else:
        #     return self.apply_standard_augmentation(im)

    def apply_standard_augmentation(
        self, im: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        fill = self.fill
        channels, height, width = F.get_dimensions(im)

        if isinstance(fill, (int, float)):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]

        """MODIFCATION: Set magnitude and remove signed"""
        magnitude = (
            float(
                magnitudes[
                    torch.randint(len(magnitudes), (1,), dtype=torch.long)
                ].item()
            )
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        # magnitude = float(magnitudes[self.severity].item())
        # if self.get_signed and op_name not in ["Solarize", "Posterize"]:
        #     magnitude *= -1.0
        """MODIFICATION: Set magnitude and remove signed"""

        # return _apply_op(
        #     img, op_name, magnitude, interpolation=self.interpolation, fill=fill
        # )

        im, aug_info = _apply_op(
            im, op_name, magnitude, fill=fill, interpolation=self.interpolation
        )
        return im, aug_info

    def apply_custom_augmentation(self, im: Tensor) -> Tuple[Tensor, List[float]]:
        augment_im, augment_info = self.apply_standard_augmentation(im)
        augmentation_type = next(iter(augment_info.keys()))
        augmentation_magnitude = augment_info[augmentation_type]
        random_crop = RandomCrop()
        confidence_aa = 1.0  # Default value

        if self.custom == False:
            # print(f"\nAugmentation info: {augment_info}\tconf: {confidence_aa}\n")
            return augment_im, [augmentation_magnitude, torch.tensor(confidence_aa)]

        if augmentation_type == "ShearX":
            # confidence_aa = comparison_metrics.gaussian(
            #     augmentation_magnitude, a=1.0, b=0.0, c=0.56, d=0.0
            # )
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            k = 2  # 2, 4
            chance = 0.224  # 0.224, 0.1
            confidence_aa = get_data(abs(augmentation_magnitude), k=k, chance=chance)
        elif augmentation_type == "ShearY":
            # confidence_aa = comparison_metrics.gaussian(
            #     augmentation_magnitude, a=1.0, b=0.02, c=0.56, d=0.0
            # )
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            k = 2  # 2, 4
            chance = 0.226  # 0.226, 0.1
            confidence_aa = get_data(abs(augmentation_magnitude), k=k, chance=chance)
        elif augmentation_type == "TranslateX":  # HVS Available
            dim1, dim2 = im.size[0], im.size[1]
            tx = augment_info[augmentation_type]
            visibility = random_crop.compute_visibility(
                dim1=dim1, dim2=dim2, tx=tx, ty=0
            )
            # k = 2
            confidence_aa = 1 - (1 - self.chance) * (1 - visibility) ** self.k
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
        elif augmentation_type == "TranslateY":  # HVS Available
            dim1, dim2 = im.size[0], im.size[1]
            ty = augment_info[augmentation_type]
            visibility = random_crop.compute_visibility(
                dim1=dim1, dim2=dim2, tx=0, ty=ty
            )
            # k = 2
            confidence_aa = 1 - (1 - self.chance) * (1 - visibility) ** self.k
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
        elif augmentation_type == "Brightness":
            # confidence_aa = comparison_metrics.sigmoid(
            #     augmentation_magnitude, 0.9753, 17.0263, -0.8297
            # )
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
            augmentation_magnitude_normalized = (augmentation_magnitude + 1.0) / 2.0
            k = 15
            chance = 0.102  # 0.102, 0.1
            if augmentation_magnitude>0.0:
                confidence_aa = 1.0
            else:
                confidence_aa = 1 - (1 - chance) * (1 - augmentation_magnitude_normalized) ** k
        elif augmentation_type == "Contrast":  # HVS Available
            # confidence_aa = comparison_metrics.sigmoid(
            #     augmentation_magnitude, 0.9914758, 13.89562814, -0.82550186
            # )
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
            augmentation_magnitude_normalized = (augmentation_magnitude + 1.0) / 2.0
            k = 15
            chance = 0.32   # 0.32, 0.1
            if augmentation_magnitude>0.0:
                confidence_aa = 1.0
            else:
                confidence_aa = 1 - (1 - chance) * (1 - augmentation_magnitude_normalized) ** k
        elif augmentation_type == "Color":
            # confidence_aa = comparison_metrics.sigmoid(
            #     augmentation_magnitude, 1.0, 4.93537641, -1.5837580
            # )
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
            augmentation_magnitude_normalized = (augmentation_magnitude + 1.0) / 2.0
            k = 15
            chance = 0.95  # 0.95, 0.1
            if augmentation_magnitude>0.0:
                confidence_aa = 1.0
            else:
                confidence_aa = 1 - (1 - chance) * (1 - augmentation_magnitude_normalized) ** k
        elif augmentation_type == "Sharpness":
            # confidence_aa = comparison_metrics.sigmoid(
            #     augmentation_magnitude, 0.9995181, 7.07685057, -1.24349678
            # )
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
            augmentation_magnitude_normalized = (augmentation_magnitude + 1.0) / 2.0
            k = 15
            chance = 0.884  # 0.95, 0.1
            if augmentation_magnitude>0.0:
                confidence_aa = 1.0
            else:
                confidence_aa = 1 - (1 - chance) * (1 - augmentation_magnitude_normalized) ** k
        elif augmentation_type == "Posterize":
            # confidence_aa = comparison_metrics.multiscale_structural_similarity(
            #     im, augment_im
            # )
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
            augmentation_magnitude_normalized = float(augmentation_magnitude // 8.0)
            k=2
            chance = 0.86
            if augmentation_magnitude_normalized==0.0:
                confidence_aa = chance
            else:
                confidence_aa = 1 - (1 - chance) * (1 - augmentation_magnitude_normalized) ** k
        elif augmentation_type == "Solarize":
            # confidence_aa = comparison_metrics.spatial_correlation_coefficient(
            #     im, augment_im
            # )
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
            augmentation_magnitude_normalized = augmentation_magnitude / 255.0
            k = 1.5
            chance = 0.512  # 0.512, 0.1
            confidence_aa = 1 - (1 - chance) * (1 - augmentation_magnitude_normalized) ** k
        elif augmentation_type == "Rotate":  # HVS Available
            # confidence_aa = comparison_metrics.gaussian(
            #     augmentation_magnitude,
            #     a=5.83337531e-01,
            #     b=-5.36740882e-03,
            #     c=2.16250254e01,
            #     d=4.16662431e-01,
            # )
            # confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
            k = 3  # 3, 4
            chance = 0.9315 # 0.9315, 0.1
            confidence_aa = get_data(abs(augmentation_magnitude) / 135.0, k=k, chance=chance)
        # elif augmentation_type == "Equalize":
        #     # confidence_aa = comparison_metrics.multiscale_structural_similarity(
        #     #     im, augment_im
        #     # )
        #     confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
        # elif augmentation_type == "AutoContrast":
        #     confidence_aa = comparison_metrics.multiscale_contrast_similarity(
        #         im, augment_im
        #     )

        """K-model for All Augmentations"""
        # if augmentation_type in [
        #     "ShearX",
        #     "ShearY",
        #     "Brightness",
        #     "Color",
        #     "Contrast",
        #     "Sharpness",
        # ]:
        #     max_magnitude = 0.99
        # elif augmentation_type in ["TranslateX", "TranslateY"]:
        #     max_magnitude = 32.0
        # elif augmentation_type == "Rotate":
        #     max_magnitude = 135.0
        # elif augmentation_type == "Posterize":
        #     max_magnitude = 8
        # elif augmentation_type == "Solarize":
        #     max_magnitude = 255.0
        # else:
        #     max_magnitude = 1.0

        # augmentation_severity = abs(
        #     int(augmentation_magnitude / max_magnitude * self.num_magnitude_bins)
        # )
        # if augmentation_type == "Solarize":
        #     augmentation_severity = self.num_magnitude_bins - augmentation_severity

        # visibility = comparison_metrics.custom_poly_common(
        #     severity=augmentation_severity, max_severity=self.num_magnitude_bins
        # )

        # # Update self.chance
        # self.chance = 0.5

        # confidence_aa = (
        #     1 - (1 - self.chance) * (1 - visibility) ** self.k
        # )  # The non-linear function
        """K-model for All Augmentations"""

        confidence_aa = torch.from_numpy(
            np.where(confidence_aa < self.chance, self.chance, confidence_aa)
        )
        # print(f'\nAugmentation info: {augment_info}\tconf: {confidence_aa}\n')
        return augment_im, [augmentation_magnitude, confidence_aa]

    def __call__(
        self, im: Optional[Image.Image]
    ) -> Optional[Tuple[Image.Image, List[float]]]:
        augment_im, augment_info = self.apply_custom_augmentation(im)
        return augment_im, augment_info

    def __repr__(self):
        s = (
            f"{self.__class__.__name__}("
            f"custom={self.custom}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )

        return s
