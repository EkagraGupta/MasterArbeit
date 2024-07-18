import torch
from torchvision import datasets, transforms
from utils.dataset import load_dataset

transform = transforms.Compose(
    [
        # transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.TrivialAugmentWide(),
    ]
)

trainloader, testloader, classes = load_dataset(batch_size=1000, transform=transform)

images, labels = next(iter(trainloader))
