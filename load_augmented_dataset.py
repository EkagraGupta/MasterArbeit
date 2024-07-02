import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from soft_augment import SoftAugment


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
    def __init__(self, dataset, custom_transform):
        self.dataset = dataset
        self.custom_transform = custom_transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.custom_transform is None:
            print("Soft cropping not applied!")
            confidence = torch.tensor(0)
        else:
            image, confidence = self.custom_transform(image)
        return image, label, confidence

    def __len__(self):
        return len(self.dataset)


def get_training_dataloader(
    mean,
    std,
    batch_size=1,
    shuffle=False,
    da=0,
    aa=0,
    length_cut=16,
    mask_cut=1,
):
    n_classes = 10
    custom_transform = None
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
        t.append(transforms.TrivialAugmentWide())

    t.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if da == 1:
        t.append(Cutout(n_holes=1, length=length_cut, mask=mask_cut))

    transform_train = transforms.Compose(t)
    training_set = datasets.CIFAR10(
        root="./data/train", train=True, download=True, transform=transform_train
    )
    # Testset with 500 samples only
    num_samples = 10
    training_set, _ = torch.utils.data.random_split(training_set,
                                                        [num_samples, len(training_set)-num_samples],
                                                        generator=torch.Generator().manual_seed(42))
    # if custom_transform is not None:
    training_set = CustomTransform(
        dataset=training_set, custom_transform=custom_transform
    )
    train_loader = torch.utils.data.DataLoader(training_set, batch_size, shuffle)
    return train_loader


def display_image(image, title=None):
    image = image / 2 + 0.5  # unnormalize
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.show()


if __name__ == "__main__":
    # below from cutout official repo
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]

    training_loader = get_training_dataloader(mean=mean, std=std, da=1, aa=1)

    for img, label, confidence in training_loader:
        for i in range(len(img)):
            display_image(
                img[i],
                title=f"Label: {label[i].item()}, Confidence: {confidence[i].item():.3f}",
            )
        break
