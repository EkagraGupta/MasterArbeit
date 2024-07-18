from torchvision.transforms import functional as F
from torch import Tensor

from utils.custom_trivial_augment import TrivialAugmentWide


def get_augmentation_info(image: Tensor) -> Tensor:
    pil_image = F.to_pil_image(image[0])
    trivial_augment = TrivialAugmentWide()
    augmented_pil_image, image_info = trivial_augment(pil_image)
    return F.to_tensor(augmented_pil_image).unsqueeze(0), image_info
