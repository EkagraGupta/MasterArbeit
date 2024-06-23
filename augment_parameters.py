from torchvision import transforms

from utils.trivial_augment import TrivialAugmentWide
from dataset import load_dataset

original_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
augmented_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomCrop(100),
            transforms.TrivialAugmentWide(),
            # transforms.Lambda(lambda x: x.mul(255).byte()),
            transforms.ToTensor(),
        ]
    )


trainloader, testloader, classes = load_dataset(batch_size=1, transform=augmented_transforme)

for i, (image, label) in enumerate(trainloader):
    augmented_image = 