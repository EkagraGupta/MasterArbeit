import torch
import torchvision
from functools import reduce
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from augmentations.trivial_augment import CustomTrivialAugmentWide
from augmentations.random_crop import RandomCrop
from augmentations.random_choice import RandomChoiceTransforms
from compute_loss import soft_loss

import random

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
        # print(f"Confidences: {confidences}\tCombined Confidence: {combined_confidence}")
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
        augmentation_magnitude = None
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
            elif isinstance(confidences, list):
                combined_confidence = confidences[1]
                augmentation_magnitude = confidences[0]
            else:
                combined_confidence = confidences

        augment_x = self.preprocess(augment_x)

        if self.robust_samples == 0:
            if augmentation_magnitude is not None:
                return augment_x, y, [augmentation_magnitude, combined_confidence]
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
    seed: Optional[int] = None
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
        transforms.RandomCrop(32, padding=4),       # For Tiny-ImageNet: 64 x 64; For CIFAR: 32 x 32
        # transforms.TrivialAugmentWide(),
        # transforms.Resize(256),
    ]

    if aggressive_augmentation:
        if custom:
            augmentations.append(
                CustomTrivialAugmentWide(
                    custom=custom,
                    augmentation_name=augmentation_name,
                    severity=augmentation_severity,
                    get_signed=augmentation_sign,
                    dataset_name=dataset_name
                ))
        else:
            augmentations.extend([transforms.TrivialAugmentWide(), transforms.ToTensor()])
        

    # custom_trivial_augment = CustomTrivialAugmentWide(
    #     custom=custom,
    #     augmentation_name=augmentation_name,
    #     severity=augmentation_severity,
    #     get_signed=augmentation_sign,
    #     dataset_name=dataset_name,
    # )
    # random_crop_augment = RandomCrop(dataset_name=dataset_name, custom=custom)

    # if aggressive_augmentation:
    #     augmentations.append(
    #         RandomChoiceTransforms(
    #             [transforms.TrivialAugmentWide(), random_crop_augment], [0.85, 0.15]
    #         )
    #     )

    if random_cropping:
        augmentations.pop(-2)  # -1, -2(if sequential)
        # for testing
        # augmentations.append(transforms.TrivialAugmentWide())
        augmentations.append(RandomCrop(dataset_name=dataset_name, custom=custom))

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
        base_trainset = datasets.CIFAR100(root="./data/train", train=True, download=True)
        base_testset = datasets.CIFAR100(root="./data/test", train=False, download=True)
    elif dataset_name == "Tiny-ImageNet":
        base_trainset = datasets.ImageFolder(root="./data/tiny-imagenet-200/new_train")
        base_testset = datasets.ImageFolder(root="./data/tiny-imagenet-200/new_test")
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


def display_image_grid(images, labels, confidences, batch_size, classes):
    """
    Displays a 5x5 grid of images with labels and confidence scores.

    Args:
        images (torch.Tensor): Batch of images.
        labels (torch.Tensor): Corresponding labels for the images.
        confidences (torch.Tensor): Corresponding confidence scores for the images.
        batch_size (int): Number of images to display in the grid (should be 25 for a 5x5 grid).
        classes (list): List of class names for labeling.
    """
    # Limit batch_size to 25 for a 5x5 grid
    batch_size = min(batch_size, 25)
    
    if isinstance(confidences, list):
        confidences = confidences[1]

    # Convert images to a grid, with 5 images per row
    grid_img = torchvision.utils.make_grid(images[:batch_size], nrow=5)

    # Convert from tensor to numpy for display
    npimg = grid_img.numpy()

    # Plot the grid with appropriate figure size (for 5x5 grid)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")

    # Add titles for each image (labels and confidence scores)
    for i in range(batch_size):
        ax = plt.subplot(5, 5, i + 1)  # Adjust to a 5x5 grid
        ax.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        ax.set_title(
            f"{labels[i].item()} ({classes[labels[i].item()]})\nConf: {confidences[i]:.2f}",
            fontsize=8
        )
        ax.axis("off")

    plt.show()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.seed(worker_seed)


if __name__ == "__main__":

    batch_size = 25
    DATASET_NAME = "CIFAR10"

    g = torch.Generator()
    g.manual_seed(1)

    transforms_preprocess, transforms_augmentation = create_transforms(
        random_cropping=True,
        aggressive_augmentation=True,
        custom=True,
        augmentation_name="Brightness",
        augmentation_severity=15,
        augmentation_sign=True,
        dataset_name=DATASET_NAME
    )
    
    print(transforms_augmentation)

    trainset, testset = load_data(
        transforms_preprocess=transforms_preprocess,
        transforms_augmentation=transforms_augmentation,
        dataset_name=DATASET_NAME
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g
    )
    classes = trainset.dataset.classes
    images, labels, confidences = next(iter(trainloader))
    display_image_grid(images, labels, confidences, batch_size=batch_size, classes=classes)
    print(f"augmentation_magnitude: {confidences[0]}\tconfidence: {confidences[1]}")

    # pil = transforms.ToPILImage()
    # im = pil(images[0])
    # im.save("./example/augmented_image_less_dark.png")


    # transforms_preprocess, transforms_augmentation = create_transforms(random_cropping=False, aggressive_augmentation=True, custom=True, dataset_name=DATASET_NAME)
    # train_path = "./data/tiny-imagenet-200/new_train"
    # custom_trainset = datasets.ImageFolder(root=train_path, transform=transforms_augmentation)
    # classes = custom_trainset.classes
    # custom_trainloader = torch.utils.data.DataLoader(custom_trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    # images, labels = next(iter(custom_trainloader))
    # confidences = np.ones((128,))
    # # images = images_data[0]
    # # confidences = images_data[1][1]
    # # print(f'Image shape: {images.shape}')
    # display_image_grid(images, labels, confidences, batch_size=batch_size, classes=classes)
    # # print(f"augmentation_magnitude: {confidences[0]}\tconfidence: {confidences[1]}")

