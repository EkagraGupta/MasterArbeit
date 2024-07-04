import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from soft_augment import SoftAugment
from trivial_augment import CustomTrivialAugmentWide

from typing import Optional, List


# official cutout implementation
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length, mask=0):
        self.n_holes = n_holes
        self.length = length
        self.mask = mask
        print("noise mask: ", str(self.mask))

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        dim1 = img.size(1)
        dim2 = img.size(2)

        # noise mix
        bg_n = torch.rand((3, dim1, dim2))
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            if self.mask:
                img[:, int(y1) : int(y2), int(x1) : int(x2)] = bg_n[
                    :, int(y1) : int(y2), int(x1) : int(x2)
                ]
            else:
                mask[int(y1) : int(y2), int(x1) : int(x2)] = 0.0

                mask = torch.from_numpy(mask)
                mask = mask.expand_as(img)
                img = img * mask

        return img


class CustomTransform(torch.utils.data.Dataset):
    """A custom dataset wrapper that applies a user-defined transformation to the images
    in the dataset.
    """

    def __init__(self, dataset, custom_transform, aggressive_augment_transform):
        """Initializes the CustomTransform class with the given dataset and custom transformation.

        Args:
            dataset (_type_): The original dataset containing images and labels.
            custom_transform (_type_): Transformation function to perform SoftCrop operation.
        """
        self.dataset = dataset
        self.custom_transform = custom_transform
        self.aggressive_augment_transform = aggressive_augment_transform

    def __getitem__(self, index: int) -> tuple:
        """Retrieves an item from the dataset at the specified index and applies the custom transformation.

        Args:
            index (int): The index of item to be retrieved.

        Returns:
            tuple: A tuple containing the transformed image, the original label, and the confidence score.
                    If custom_transform is None, the original image and a confidence score of 0.0 are returned.
        """
        image, label = self.dataset[index]
        confidence = 1.0
        augment_info = {"None": 1.0}

        if self.aggressive_augment_transform is not None:
            image, augment_info = self.aggressive_augment_transform(image)
            # confidence = augment_info.values()
        if self.custom_transform is not None:
            image, confidence = self.custom_transform(image, augment_info)
        print(f'Augmentation Info: {augment_info}')
        return image, label, confidence

    def __len__(self):
        """Returns the total number of items in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset)


def get_dataloader(
    num_samples: Optional[int],
    batch_size: int = 1,
    shuffle: bool = False,
    da: int = 0,
    aa: int = 0,
    length_cut: int = 16,
    mask_cut: Optional[int] = 1,
    # normalize: bool = True,
    train: bool = True,
) -> torch.utils.data.DataLoader:
    """Creates and returns a DataLoader for training with specified data augmentations.

    Args:
        batch_size (int, optional): The number of samples per batch. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        da (int, optional): Data augmentation mode. Defaults to 0.
            - -1: No data augmentation.
            - 0: Standard data augmentation (RandomCrop and RandomHorizontalFlip).
            - 1: Standard data augmentation with cutout.
            - 2: RandomHorizontalFlip with SoftCrop data augmentation.
        aa (int, optional): Aggressive augmentation mode. Defaults to 0.
            - -1: No aggressive augmentation.
            - 0: Random augmentation.
            - 1: Trivial augmentation.
        length_cut (int, optional): The length for the cutout. Defaults to 16.
        mask_cut (Optional[int], optional): The mask value for the cutout. Defaults to 1.
        normalize (bool): Whether to define mean and standard deviation to normalize images.
        num_samples (int, optional): Number of samples to be included in truncated dataset (for testing).
        train (bool): Whether to get training loader or test loader.

    Returns:
        torch.utils.data.DataLoader: Instance for the training dataset.
    """
    n_classes = 10
    aa_transform, custom_transform = None, None

    # if normalize:
    #     # below from cutout official repo
    #     mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    #     std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    # else:
    #     mean = [0.0 for _ in range(3)]
    #     std = [1.0 for _ in range(3)]

    # Define data augmentation transformations
    if da == -1:
        print("No data augmentation!")
        t = []
    elif da == 0:  # original augmentaiton
        print("Standard data augmentation!")
        t = [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    elif da == 1:  # cutout will be appended to the end
        print("Standard data augmentation with cutout!")
        t = [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    elif da == 2:
        print("RandomFlip + SoftCrop data augmentation!")
        t = [
            transforms.RandomHorizontalFlip(),
        ]
        custom_transform = SoftAugment(n_class=n_classes, k=2)

    if aa == -1:
        print("No Aggressive augmentations applied!\n")
    elif aa == 0:
        print("Using Random Augmentation!\n")
        t.append(transforms.RandAugment())
    elif aa == 1:
        print("Using Trivial Augmentation!\n")
        # t.append(CustomTrivialAugmentWide())
        aa_transform = CustomTrivialAugmentWide()

    # Add standard transformations
    # t.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    t.extend([transforms.ToTensor()])

    # Add cutout if specified
    if da == 1:
        t.append(Cutout(n_holes=1, length=length_cut, mask=mask_cut))

    # Compose all transformations
    transform = transforms.Compose(t)

    if train:
        # Load CIFAR-10 training dataset with transformations
        data_set = datasets.CIFAR10(
            root="./data/train", train=True, download=True, transform=transform
        )
    else:
        # Load CIFAR-10 test dataset with transformations
        data_set = datasets.CIFAR10(
            root="./data/test", train=False, download=True, transform=transform
        )

    if num_samples:
        # Split dataset to create a smaller subset (for testing)
        data_set, _ = torch.utils.data.random_split(
            data_set,
            [num_samples, len(data_set) - num_samples],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"\nTruncated dataset to {len(data_set)} images.\n")

    # Apply custom transformation (will be ineffective if not specified)
    data_set = CustomTransform(
        dataset=data_set,
        custom_transform=custom_transform,
        aggressive_augment_transform=aa_transform,
    )

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(data_set, batch_size, shuffle)
    return train_loader


def display_image(image: torch.Tensor, title: Optional[str] = None) -> None:
    """Displays an image tensor

    Args:
        image (torch.Tensor): The image tensor to display.
        title (Optional[str], optional): The title of the image. Defaults to None.
    """
    # image = image / 2 + 0.5  # unnormalize
    np_image = image.numpy()
    np_image = np.clip(np_image, 0, 1)
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
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
    training_loader = get_dataloader(
        da=2, aa=1, num_samples=10, shuffle=True, train=False
    )

    for img, label, confidence in training_loader:
        for i in range(len(img)):
            display_image(
                img[i],
                title=f"Label: {label[i].item()} ({classes[label[i].item()]}) - Confidence: {confidence[i].item():.3f}",
            )
