from torchvision import transforms
from torchvision.transforms import functional as F
from torch import Tensor

from utils.trivial_augment import TrivialAugmentWide
from dataset import load_dataset

original_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)
augmented_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomCrop(100),
        # transforms.TrivialAugmentWide(),
        # transforms.ToTensor(),
    ]
)


trainloader, testloader, classes = load_dataset(
    batch_size=1, transform=original_transform
)


def get_augmentation_info(image: Tensor) -> Tensor:
    pil_image = F.to_pil_image(image[0])
    augmented_pil_image, image_info = TrivialAugmentWide(pil_image)
    return F.to_tensor(augmented_pil_image).unsqueeze(0)
