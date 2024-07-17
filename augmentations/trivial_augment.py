import torch
from torchvision import transforms
from utils.trivial_augment_wide import TrivialAugmentWide

from utils.dataset import load_dataset
from utils.custom_trivial_augment import CTrivialAugmentWide
from utils.sift_comparison import sift_correction_factor
from utils.vif import vif


class CustomTrivialAugmentWide:

    def __init__(self, custom: bool = False):
        self.custom = custom

    def __call__(self, im: torch.tensor) -> tuple:
        if self.custom:
            augment_im, augment_info = self.get_augment_info(im)
            return augment_im, augment_info
        else:
            trivial_augmentation = TrivialAugmentWide()
            augment_im = trivial_augmentation(im)
            return augment_im

    @staticmethod
    def get_augment_info(im: torch.tensor) -> tuple:
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
        augmentation_type = next(iter(im_info.keys()))
        if augmentation_type in pixelwise_augs:
            vif_value = vif(original_image=im, augmented_image=augment_im)
            confidence_aa = vif_value.item()
        else:
            confidence_aa = sift_correction_factor(
                original_image=im, augmented_image=augment_im
            )
        print(f"\nAugmentation info: {im_info}\n")
        return augment_im, confidence_aa


if __name__ == "__main__":
    from utils.dataset import load_dataset

    transform = transforms.Compose(
        [transforms.ToTensor(), CustomTrivialAugmentWide(custom=True)]
    )
    trainloader, testloader, classes = load_dataset(batch_size=1, transform=transform)
    data = next(iter(trainloader))
    print(len(data[0]))
