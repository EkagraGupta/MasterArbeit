import torch
import torchvision
from functools import reduce
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from augmentations.trivial_augment import CombinedTrivialAugmentWide
from augmentations.random_crop import RandomCrop
from compute_loss import soft_loss


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions.

    Attributes:
        dataset (torch.utils.data.Dataset): The base dataset to augment.
        transforms_preprocess (transforms.Compose): Transformations for preprocessing.
        transforms_augmentation (transforms.Compose): Transformations for augmentation.
        transforms_generated (transforms.Compose): Transformations for generated samples.
        robust_samples (int): Number of robust samples to include.
    """

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
        elif dataset.__class__.__name__ == "CIFAR10":
            # CIFAR-10
            self.dataset = datasets.CIFAR10(
                root="./data/train", train=True, download=True
            )
        elif dataset.__class__.__name__ == "CIFAR100":
            # CIFAR-100
            self.dataset = datasets.CIFAR100(
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

    def get_confidence(self, confidences: Optional[tuple]) -> Optional[float]:
        """Combines multiple confidence values into a single value.

        Args:
            confidences (Optional[tuple]): A tuple of confidence values.

        Returns:
            Optional[float]: The combined confidence value.
        """
        combined_confidence = reduce(lambda x, y: x * y, confidences)
        # print(f'Combined confidence: {combined_confidence}')
        return combined_confidence

    def __getitem__(self, i: Optional[int]) -> Optional[tuple]:
        """Retrieves an item from the dataset and applies augmentations.

        Args:
            i (Optional[int]): Index of the item to retrieve.

        Returns:
            Optional[tuple]: The augmented image, the label, and the combined confidence value.
        """
        x, y = self.dataset[i]
        confidences = None
        combined_confidence = torch.tensor(1.0, dtype=torch.float32)
        original = True  # for now "original" is set to True rather than returning from base_dataset

        augment = (
            self.transforms_augmentation
            if original == True
            else self.transforms_generated
        )

        augment_x = augment(x)

        if isinstance(augment_x, tuple):
            confidences = augment_x[1]
            augment_x = augment_x[0]
            if isinstance(confidences, tuple):
                combined_confidence = self.get_confidence(confidences)
            else:
                combined_confidence = confidences

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
    random_cropping: bool = False,
    aggressive_augmentation: bool = False,
    custom: bool = False,
    augmentation_name: str = None,
    augmentation_severity: int = 0,
    augmentation_sign: bool = False,
    dataset_name: str = "CIFAR10",
) -> Optional[tuple]:
    """Creates preprocessing and augmentation transformations.

    Args:
        random_cropping (bool, optional): Flag to include random cropping in augmentations. Defaults to False.
        aggressive_augmentation (bool, optional): Flag to include aggressive augmentations. Defaults to False.
        custom (bool, optional): Flag to use custom trivial augmentations. Defaults to False.

    Returns:
        Optional[tuple]: The preprocessing and augmentation transformations.
    """
    t = [transforms.ToTensor()]
    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    ]

    if aggressive_augmentation:
        # augmentations.append(
        #     CustomTrivialAugmentWide(
        #         custom=custom,
        #         augmentation_name=augmentation_name,
        #         severity=augmentation_severity,
        #         get_signed=augmentation_sign,
        #         dataset_name=dataset_name
        #     )
        augmentations.append(
            CombinedTrivialAugmentWide(
                custom=custom,
                augmentation_name=augmentation_name,
                severity=augmentation_severity,
                get_signed=augmentation_sign,
                dataset_name=dataset_name,
            )
        )

    if random_cropping:
        augmentations.append(RandomCrop(dataset_name=dataset_name))

    transforms_preprocess = transforms.Compose(t)
    transforms_augmentation = transforms.Compose(augmentations)

    return transforms_preprocess, transforms_augmentation


def load_data(
    transforms_preprocess,
    transforms_augmentation=None,
    dataset_split: Optional[int] = "full",
    dataset_name: Optional[str] = "CIFAR10",
) -> Optional[tuple]:
    """Loads and prepares the CIFAR-10 dataset with specified transformations.

    Args:
        transforms_preprocess (transforms.Compose): Preprocessing transformations.
        transforms_augmentation (transforms.Compose, optional): Augmentation transformations.
        robust_samples (int, optional): Number of robust samples to include.

    Returns:
        Optional[tuple]: The training and testing datasets.
    """
    if dataset_name == "CIFAR10":
        # CIFAR-10
        base_trainset = datasets.CIFAR10(root="./data/train", train=True, download=True)
        base_testset = datasets.CIFAR10(root="./data/test", train=False, download=True)
    elif dataset_name == "CIFAR100":
        # CIFAR-100
        base_trainset = datasets.CIFAR100(
            root="./data/train", train=True, download=True
        )
        base_testset = datasets.CIFAR100(root="./data/test", train=False, download=True)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    """MODIFICATION: Truncate the dataset to a smaller size for faster testing"""
    if dataset_split != "full":
        truncated_trainset = torch.utils.data.Subset(
            base_trainset, range(dataset_split)
        )
        truncated_testset = torch.utils.data.Subset(base_testset, range(dataset_split))
    else:
        truncated_trainset = base_trainset
        truncated_testset = base_testset
    """MODIFICATION: Truncate the dataset to a smaller size for faster testing"""

    if transforms_augmentation is not None:
        trainset = AugmentedDataset(
            dataset=truncated_trainset,
            transforms_preprocess=transforms_preprocess,
            transforms_augmentation=transforms_augmentation,
        )

        testset = AugmentedDataset(
            dataset=truncated_testset,
            transforms_preprocess=transforms_preprocess,
            transforms_augmentation=transforms_augmentation,
        )
    elif base_trainset.__class__.__name__ == "CIFAR10":
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
    elif base_trainset.__class__.__name__ == "CIFAR100":
        trainset = datasets.CIFAR100(
            root="./data/train",
            train=True,
            download=True,
            transform=transforms_preprocess,
        )
        testset = datasets.CIFAR100(
            root="./data/test",
            train=False,
            transform=transforms_preprocess,
            download=True,
        )

    return trainset, testset


def display_image_grid(images, labels, confidences, batch_size):
    """
    Displays a grid of images with labels and confidence scores.

    Args:
        images (torch.Tensor): Batch of images.
        labels (torch.Tensor): Corresponding labels for the images.
        confidences (torch.Tensor): Corresponding confidence scores for the images.
        batch_size (int): Number of images to display in the grid.
    """
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

    if isinstance(confidences, list):
        confidences = confidences[1]

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
    plt.show()


if __name__ == "__main__":
    batch_size = 1000

    transforms_preprocess, transforms_augmentation = create_transforms(
        random_cropping=False,
        aggressive_augmentation=True,
        custom=True,
        augmentation_name="Rotate",
        augmentation_severity=30,
        augmentation_sign=False,
    )
    trainset, testset = load_data(
        transforms_preprocess=transforms_preprocess,
        transforms_augmentation=transforms_augmentation,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False
    )
    images, labels, confidences = next(iter(trainloader))
    # display_image_grid(images, labels, confidences, batch_size=batch_size)
    print(f"Confidence: {confidences}")
    
    print(len(confidences[1]))

    # compute loss
    # loss = soft_loss(images, labels, confidences)
    # print(f'Loss: {loss}')
