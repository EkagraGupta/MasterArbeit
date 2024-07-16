import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from augmentations.random_crop import RandomCrop
from augmentations.trivial_augment import CustomTrivialAugmentWide
from typing import Optional, List


class Cutout(object):
    """Randomly mask out one or more patches from an image."""

    def __init__(self, n_holes: int, length: int, mask: int = 0):
        self.n_holes = n_holes
        self.length = length
        self.mask = mask
        print("Noise mask:", str(self.mask))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        bg_n = torch.rand((3, h, w))

        for _ in range(self.n_holes):
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

        if not self.mask:
            mask = torch.from_numpy(mask).expand_as(img)
            img = img * mask

        return img


class CustomTransform(torch.utils.data.Dataset):
    """A custom dataset wrapper that applies a user-defined transformation
    to the images."""

    def __init__(self, dataset, custom_transform, aggressive_augment_transform):
        self.dataset = dataset
        self.custom_transform = custom_transform
        self.aggressive_augment_transform = aggressive_augment_transform

    def __getitem__(self, index: int) -> tuple:
        im, label = self.dataset[index]
        confidence = 1.0
        aa_info = {"None": 1.0}

        if self.aggressive_augment_transform is not None:
            im, aa_info = self.aggressive_augment_transform(im)
            confidence = next(iter(aa_info.values()))
        if self.custom_transform is not None:
            im, confidence = self.custom_transform(im, aa_info)
        return im, label, confidence

    def __len__(self):
        return len(self.dataset)


def get_dataloader(
    num_samples: Optional[int],
    batch_size: int = 1,
    shuffle: bool = False,
    da: int = 0,
    aa: int = 1,
    length_cut: int = 16,
    mask_cut: Optional[int] = 1,
    train: bool = True,
) -> torch.utils.data.DataLoader:
    n_classes = 10
    aa_transform, custom_transform = None, None

    if da == -1:
        t = []
    elif da == 0:
        t = [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    elif da == 1:
        t = [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    elif da == 2:
        t = [transforms.RandomHorizontalFlip()]
        custom_transform = RandomCrop(n_class=n_classes, k=2)

    if aa == -1:
        pass
    elif aa == 0:
        t.append(transforms.RandAugment())
    elif aa == 1:
        aa_transform = CustomTrivialAugmentWide()

    t.append(transforms.ToTensor())
    if da == 1:
        t.append(Cutout(n_holes=1, length=length_cut, mask=mask_cut))

    transform = transforms.Compose(t)

    dataset = datasets.CIFAR10(
        root="./data/train" if train else "./data/test",
        train=train,
        download=True,
        transform=transform,
    )
    if num_samples:
        dataset, _ = torch.utils.data.random_split(
            dataset,
            [num_samples, len(dataset) - num_samples],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"Truncated dataset to {len(dataset)} images.")

    dataset = CustomTransform(
        dataset=dataset,
        custom_transform=custom_transform,
        aggressive_augment_transform=aa_transform,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def display_image(image: torch.Tensor, title: Optional[str] = None) -> None:
    im_numpy = image.numpy()
    im_numpy = np.clip(im_numpy, 0, 1)
    plt.imshow(np.transpose(im_numpy, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def display_images_in_grid(
    images: List[torch.Tensor],
    labels: List[int],
    confidences: List[float],
    classes: List[str],
) -> None:
    """Displays a list of image tensors in a grid.

    Args:
        images (List[torch.Tensor]): The list of image tensors to display.
        labels (List[int]): The list of labels for the images.
        confidence (List[float]): The list of confidence scores for the images.
        classes (List[str]): The list of class names.
    """
    _, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for idx, (im, label, confidence) in enumerate(zip(images, labels, confidences)):
        im_numpy = im.numpy()
        im_numpy = np.clip(im_numpy, 0, 1)
        axes[idx].imshow(np.transpose(im_numpy, (1, 2, 0)))
        axes[idx].set_title(f"{classes[label]} ({confidence: .3f})")
        axes[idx].axis("off")

    plt.tight_layout()
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
        da=2, aa=1, num_samples=100, shuffle=False, train=False
    )

    images, labels, confidences = [], [], []
    for im, label, confidence in training_loader:
        # for i in range(len(img)):
        #     display_image(
        #         img[i],
        #         title=f"Label: {label[i].item()} ({classes[label[i].item()]}) - Confidence: {confidence[i].item():.3f}",
        #     )
        images.extend(im)
        labels.extend(label)
        confidences.extend(np.clip(confidence, 0.0, 1.0))
    display_images_in_grid(images, labels, confidences, classes)
