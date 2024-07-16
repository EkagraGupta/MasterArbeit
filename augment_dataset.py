import torch
from functools import reduce

from augmentations.trivial_augment import CustomTrivialAugmentWide
from augmentations.random_crop import RandomCrop


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions."""

    def __init__(
        self,
        dataset,
        # transforms_preprocess,
        transforms_augmentation,
        transforms_generated=None,
        robust_samples=0,
    ):
        self.dataset = dataset
        # self.preprocess = transforms_preprocess
        self.transforms_augmentation = transforms_augmentation
        self.transforms_generated = (
            transforms_generated if transforms_generated else transforms_augmentation
        )
        self.robust_samples = robust_samples
        self.original_length = getattr(dataset, "original_length", None)
        self.generated_length = getattr(dataset, "generated_length", None)

    def get_confidence(self, confidences):
        return reduce(lambda x, y: x * y, confidences)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        # augment = (
        #     self.transforms_augmentation
        #     if original == True
        #     else self.transforms_generated
        # )
        augmented_x, confidences = self.transforms_augmentation(x)
        combined_confidence = self.get_confidence(confidences)
        # if self.robust_samples == 0:
        #     return augment(x), y
        # elif self.robust_samples == 1:
        #     im_tuple = (self.preprocess(x), augment(x))
        #     return im_tuple, y
        # elif self.robust_samples == 2:
        #     im_tuple = (self.preprocess(x), augment(x), augment(x))
        #     return im_tuple, y
        return augmented_x, y, combined_confidence

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":

    from torchvision import datasets, transforms
    from PIL import Image

    transforms_preprocess = transforms.Compose([transforms.ToTensor()])
    transforms_randomcrop = RandomCrop()
    transforms_aggressiveaugment = CustomTrivialAugmentWide(custom=True)
    transform_combined = transforms.Compose(
        [transforms.ToTensor(), transforms_aggressiveaugment, transforms_randomcrop]
    )
    base_dataset = datasets.CIFAR10(root="./data/train", train=True, download=True)

    augmentation_x = AugmentedDataset(
        transforms_augmentation=transform_combined, dataset=base_dataset
    )

    trainloader = torch.utils.data.DataLoader(
        augmentation_x, batch_size=1, shuffle=False
    )

    images, labels, confidences = next(iter(trainloader))
    print(confidences)
