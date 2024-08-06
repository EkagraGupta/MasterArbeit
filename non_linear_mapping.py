from augment_dataset import create_transforms, load_data
from utils.plot_non_linear_curve import plot_mean_std, get_mean_std
import torch

augmentation_types = [
    "Identity",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Rotate",
    "Brightness",
    "Color",
    "Contrast",
    "Sharpness",
    "Posterize",
    "Solarize",
    "AutoContrast",
    "Equalize",
]


def get_plot(augmentation_type, dataset_split=100):
    mean_list, std_list = [], []

    for severity in range(1, 30):
        print(f"Processing severity: {severity}\n")
        preprocess, augmentation = create_transforms(
            random_cropping=False,
            aggressive_augmentation=True,
            custom=True,
            augmentation_name=augmentation_type,
            augmentation_severity=severity,
        )
        trainset, _ = load_data(
            transforms_preprocess=preprocess,
            transforms_augmentation=augmentation,
            dataset_split=dataset_split,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset,
            shuffle=False,
            batch_size=dataset_split,
        )
        _, _, confidences = next(iter(trainloader))
        mean, std = get_mean_std(confidences)
        mean_list.append(mean.item())
        std_list.append(std.item())

    plot_mean_std(mean_list, std_list, augmentation_type)


for augmentation_type in augmentation_types:
    get_plot(augmentation_type)

# print(f'Mean: {mean_list}\tStd: {std_list}')
