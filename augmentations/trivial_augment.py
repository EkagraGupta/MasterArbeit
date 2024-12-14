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

from torchvision import transforms

from utils.sift_comparison import sift_correction_factor
from utils.orb_comparison import orb_correction_factor
from utils.vif import compute_vif


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
        self.dataset_name = dataset_name

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
        elif dataset_name == "Tiny-ImageNet":
            self.chance = 1 / 200
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
        # if magnitudes.ndim > 0:
        #     magnitude = float(magnitudes[self.severity].item())
        # else:
        #     magnitude = 0.0
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
        
        """Performance data obtained from available HVS"""
        occlusion_hvs = [0.216, 0.388, 0.51066667, 0.584, 0.65333333, 0.68533333, 0.68, 0.72666667, 0.75466667, 0.764, 0.776, 0.78758974, 0.79876923, 0.80994872, 0.82112821, 0.83230769, 0.84348718, 0.85466667, 0.86584615, 0.87702564, 0.88820513, 0.89938462, 0.9105641, 0.92174359, 0.93292308, 0.94410256, 0.95528205, 0.96646154, 0.97764103, 0.98882051, 1.]
        rotation_hvs = [1., 0.9985, 0.997, 0.9955, 0.994, 0.9925, 0.991, 0.9895, 0.988, 0.9865, 0.985, 0.9835, 0.982, 0.9805, 0.979, 0.9775, 0.976, 0.9745, 0.973, 0.9715, 0.97, 0.964, 0.958, 0.952, 0.946, 0.94, 0.934, 0.9315, 0.936, 0.9405, 0.945]
        contrast_hvs = [0.32, 0.32, 0.64254054, 0.96603963, 0.96734732, 0.96865501, 0.9699627, 0.9712704, 0.97257809, 0.97388578, 0.97519347, 0.97650117, 0.97780886, 0.97911655, 0.98042424, 0.98173193, 0.98303963, 0.98434732, 0.98565501, 0.98696271, 0.9882704, 0.98957809, 0.99088578, 0.99219347, 0.99350117, 0.99480886, 0.99611655, 0.99742424, 0.99873194, 1., 1.]
        """Performance data obtained from available HVS"""

        dat = self._augmentation_space(self.num_magnitude_bins)

        if augmentation_type in ['Identity', 'AutoContrast', 'Equalize', 'Invert']:
            augmentation_idx = 0
        else:
            mags = dat[augmentation_type]
            for i in range(len(mags[0])):
                if round(abs(augmentation_magnitude), 5) == round(mags[0][i].item(), 5):
                    augmentation_idx = i
                    break

            
        if augmentation_type == "ShearX":
            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            """Mapping function from Rotation HVS"""
            k = 3  # 1.5, 3
            chance = 0.224  # 0.224, 0.1
            confidence_aa = 1 - (1 - self.chance) * abs(augmentation_magnitude) ** self.k

            """Mapping function from Translation HVS"""
            # dim1, dim2 = im.size[0], im.size[1]
            # visibility = random_crop.compute_visibility(
            #     dim1=dim1, dim2=dim2, tx=0., ty=augmentation_magnitude
            # )
            # k = 2
            # chance = 0.224          # taken from model acc
            # confidence_aa = 1 - (1 - chance) * (1 - visibility) ** k

            """Exact Rotation HVS"""
            # confidence_aa = rotation_hvs[augmentation_idx]

        elif augmentation_type == "ShearY":
            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            """Mapping function from Rotation HVS"""
            k = 3  # 1.5, 3
            chance = 0.226  # 0.226, 0.1
            confidence_aa = 1 - (1 - self.chance) * abs(augmentation_magnitude) ** self.k

            """Mapping function from Translation HVS"""
            # dim1, dim2 = im.size[0], im.size[1]
            # visibility = random_crop.compute_visibility(
            #     dim1=dim1, dim2=dim2, tx=0., ty=augmentation_magnitude
            # )
            # k = 2
            # chance = 0.224
            # confidence_aa = 1 - (1 - chance) * (1 - visibility) ** k

            """Exact Rotation HVS"""
            # confidence_aa = rotation_hvs[augmentation_idx]

        elif augmentation_type == "TranslateX":  # HVS Available
            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            """Mapping function from Translation HVS"""
            dim1, dim2 = im.size[0], im.size[1]
            visibility = random_crop.compute_visibility(
                dim1=dim1, dim2=dim2, tx=augmentation_magnitude, ty=0
            )
            k = 4               # 2, 4
            chance = 0.216        # 0.102, 0.216 
            confidence_aa = 1 - (1 - self.chance) * (1 - visibility) ** self.k

            """Exact Occlusion HVS"""
            # confidence_aa = occlusion_hvs[::-1][augmentation_idx]

        elif augmentation_type == "TranslateY":  # HVS Available
            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
            
            """Mapping function from Translation HVS"""
            dim1, dim2 = im.size[0], im.size[1]
            visibility = random_crop.compute_visibility(
                dim1=dim1, dim2=dim2, tx=0, ty=augmentation_magnitude
            )
            k = 4               # 2, 4
            chance = 0.216        # 0.102, 0.216
            confidence_aa = 1 - (1 - self.chance) * (1 - visibility) ** self.k

            """Exact Occlusion HVS"""
            # confidence_aa = occlusion_hvs[::-1][augmentation_idx]

        elif augmentation_type == "Brightness":

            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            """Mapping function from Contrast HVS"""
            # k_neg, k_pos = 20, 3                # (3, 2), (20, 3) 
            # chance_pos = 0.86                   # model_acc[-1]
            # chance_neg = 0.32                   # 0.102, 0.32
            # if augmentation_magnitude>0.0:
            #     confidence_aa = 1 - (1 - chance_pos) * (augmentation_magnitude) ** k_pos
            # else:
            #     confidence_aa = 1 - (1 - chance_neg) * (abs(augmentation_magnitude)) ** k_neg
            if augmentation_magnitude>0.0:
                confidence_aa = 1.0
            else:
                confidence_aa = 1 - (1 - self.chance) * (abs(augmentation_magnitude)) ** self.k


            """Exact Contrast HVS"""
            # if augmentation_magnitude>0.0:
            #     confidence_aa = 1.0
            # else:
            #     confidence_aa = contrast_hvs[::-1][augmentation_idx]

        elif augmentation_type == "Contrast":  # HVS Available

            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            """Mapping function from Contrast HVS"""
            k_neg, k_pos = 20, 3                # (3, 2), (20, 3) 
            chance_pos = 0.976                   # model_acc[-1]
            chance_neg = 0.32                   # 0.102, 0.32
            if augmentation_magnitude>0.0:
                # confidence_aa = 1 - (1 - chance_pos) * (augmentation_magnitude) ** k_pos
                confidence_aa = 1.0
            else:
                confidence_aa = 1 - (1 - self.chance) * (abs(augmentation_magnitude)) ** self.k

            """Exact Contrast HVS"""
            # if augmentation_magnitude>0.0:
            #     confidence_aa = 1.0
            # else:
            #     confidence_aa = contrast_hvs[::-1][augmentation_idx]

        elif augmentation_type == "Color":

            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            """Mapping function from Model Accuracy"""
            k = 5                       # 2, 5   
            chance = 0.95               # 0.1, 0.95   
            if augmentation_magnitude>0.0:
                confidence_aa = 1.0
            else:
                confidence_aa = 1 - (1 - self.chance) * (abs(augmentation_magnitude)) ** self.k

            """Exact Contrast HVS"""
            # if augmentation_magnitude>0.0:
            #     confidence_aa = 1.0
            # else:
            #     confidence_aa = contrast_hvs[::-1][augmentation_idx]

        elif augmentation_type == "Sharpness":

            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            """Mapping function from Model Accuracy"""
            k = 7                       # 2, 7   
            chance = 0.884               # 0.1, 0.884   
            if augmentation_magnitude>0.0:
                confidence_aa = 1.0
            else:
                confidence_aa = 1 - (1 - self.chance) * (abs(augmentation_magnitude)) ** self.k

            """Exact Contrast HVS"""
            # if augmentation_magnitude>0.0:
            #     confidence_aa = 1.0
            # else:
            #     confidence_aa = contrast_hvs[::-1][augmentation_idx]

        elif augmentation_type == "Posterize":
            """Image Similarity Metric"""
            # confidence_aa = comparison_metrics.multiscale_structural_similarity(
            #     im, augment_im
            # )

            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            """Mapping function from Model Accuracy"""
            augmentation_magnitude_normalized = float(augmentation_magnitude / 8.0)
            k = 2           # 1.5, 2
            chance = 0.86   # 0.1, 0.86
            confidence_aa = 1 - (1 - self.chance) * (1 - augmentation_magnitude_normalized) ** self.k

        elif augmentation_type == "Solarize":
            """Image Similarity Metric"""
            # confidence_aa = comparison_metrics.spatial_correlation_coefficient(
            #     im, augment_im
            # )

            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            """Mapping function from Model Accuracy"""
            augmentation_magnitude_normalized = augmentation_magnitude / 255.0
            k = 2           # 1.5, 2
            chance = 0.512  # 0.1, 0.512
            confidence_aa = 1 - (1 - self.chance) * (1 - augmentation_magnitude_normalized) ** self.k

        elif augmentation_type == "Rotate":  # HVS Available

            """Exact Model Accuracy"""
            # confidence_aa, _ = model_accuracy_mapping(augmentation_magnitude, augmentation_type)

            """Mapping function from Rotation HVS"""
            k = 3  # 2, 3
            chance = 0.9315 # 0.2, 0.9315
            confidence_aa = 1 - (1 - self.chance) * (abs(augmentation_magnitude) / 135.0) ** self.k

            """Exact Rotation HVS"""
            # confidence_aa = rotation_hvs[augmentation_idx]

        # elif augmentation_type == "Equalize":
        #     # confidence_aa = comparison_metrics.multiscale_structural_similarity(
        #     #     im, augment_im
        #     # )
        #     confidence_aa = model_accuracy_mapping(augmentation_magnitude, augmentation_type)
        # elif augmentation_type == "AutoContrast":
        #     confidence_aa = comparison_metrics.multiscale_contrast_similarity(
        #         im, augment_im
        #     )

        # confidence_aa = torch.from_numpy(
        #     np.where(confidence_aa < 0.5, 0.5, confidence_aa)
        # )

        if self.dataset_name=="Tiny-ImageNet":
            to_tensor = transforms.Compose([transforms.ToTensor()])
            augment_im = to_tensor(augment_im)

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
