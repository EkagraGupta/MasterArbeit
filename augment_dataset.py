import torch
import torchvision
from functools import reduce
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from augmentations.trivial_augment import CustomTrivialAugmentWide
from augmentations.random_crop import RandomCrop


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions."""

    def __init__(
        self,
        dataset,
        transforms_preprocess,
        transforms_augmentation,
        transforms_generated=None,
        robust_samples=0,
    ):
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = datasets.CIFAR10(
                root="./data/train", train=True, download=True
            )
        self.preprocess = transforms_preprocess
        self.transforms_augmentation = transforms_augmentation
        self.transforms_generated = (
            transforms_generated if transforms_generated else transforms_augmentation
        )
        self.robust_samples = robust_samples
        self.original_length = getattr(dataset, "original_length", None)
        self.generated_length = getattr(dataset, "generated_length", None)

    def get_confidence(self, confidences):
        combined_confidence = reduce(lambda x, y: x * y, confidences)
        return combined_confidence

    def __getitem__(self, i):
        x, y = self.dataset[i]
        confidences = None
        combined_confidence = torch.tensor(1.0, dtype=torch.float32)
        # for now "original" is set to True rather than returning from base_dataset
        original = True
        augment = (
            self.transforms_augmentation
            if original == True
            else self.transforms_generated
        )
        augment_x = self.transforms_augmentation(x)
        if isinstance(augment_x, tuple):
            confidences = augment_x[1]
            augment_x = augment_x[0]
            if isinstance(confidences, tuple):

                combined_confidence = self.get_confidence(confidences)
            else:
                combined_confidence = confidences
        # print(f'confidences: {confidences}\tcombined_confidence: {combined_confidence}\nType: {type(combined_confidence)}')
        augment_x = self.preprocess(augment_x)
        if self.robust_samples == 0:
            return augment_x, y, combined_confidence
        # elif self.robust_samples == 1:
        #     im_tuple = (self.preprocess(x), augment(x))
        #     return im_tuple, y
        # elif self.robust_samples == 2:
        #     im_tuple = (self.preprocess(x), augment(x), augment(x))
        #     return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def create_transforms(
    random_cropping=False, aggressive_augmentation=False, custom=False
):
    t = [transforms.ToTensor()]
    augmentations = []

    if aggressive_augmentation:
        augmentations.append(CustomTrivialAugmentWide(custom=custom))
    if random_cropping:
        augmentations.append(RandomCrop())
    # augmentations.append(transforms.ToTensor())
    transforms_preprocess = transforms.Compose(t)
    transforms_augmentation = transforms.Compose(augmentations)

    return transforms_preprocess, transforms_augmentation


def load_data(transforms_preprocess, transforms_augmentation=None, robust_samples=0):

    base_trainset = datasets.CIFAR10(root="./data/train", train=True, download=True)
    base_testset = datasets.CIFAR10(root="./data/test", train=False, download=True)

    if transforms_augmentation is not None:
        trainset = AugmentedDataset(
            dataset=base_trainset,
            transforms_preprocess=transforms_preprocess,
            transforms_augmentation=transforms_augmentation,
        )

        testset = AugmentedDataset(
            dataset=base_testset,
            transforms_preprocess=transforms_preprocess,
            transforms_augmentation=transforms_augmentation,
        )
    else:
        trainset = datasets.CIFAR10(
            root="./data/train",
            train=True,
            download=True,
            transform=transforms_preprocess,
        )
        testset = datasets.CIFAR10(
            root="./data/test",
            train=False,
            transform=transforms_preprocess,
            download=True,
        )
    return trainset, testset


def display_image_grid(images, labels, confidences, batch_size):
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Convert images to a grid
    grid_img = torchvision.utils.make_grid(images, nrow=batch_size)

    # Convert from tensor to numpy for display
    npimg = grid_img.numpy()

    # Plot the grid
    plt.figure(figsize=(batch_size * 2, 2))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")

    # Add titles
    for i in range(batch_size):
        ax = plt.subplot(1, batch_size, i + 1)
        ax.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        ax.set_title(
            f"{labels[i].item()} ({classes[labels[i].item()]})\nConf: {confidences[i]:.2f}"
        )
        ax.axis("off")
    plt.suptitle("Random Cropping + Trivial Augment Applied!")
    plt.show()


if __name__ == "__main__":
    batch_size = 10
    base_dataset = datasets.CIFAR10(root="./data/train", train=True, download=True)

    transforms_preprocess, transforms_augmentation = create_transforms(
        random_cropping=False, aggressive_augmentation=True, custom=True
    )
    trainset, testset = load_data(
        transforms_preprocess=transforms_preprocess,
        transforms_augmentation=transforms_augmentation,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False
    )

    images, labels, confidences = next(iter(trainloader))
    display_image_grid(images, labels, confidences, batch_size=batch_size)
