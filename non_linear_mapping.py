from augment_dataset import create_transforms, load_data
from utils.plot_non_linear_curve import plot_mean_std, get_mean_std
from evaluate import evaluate_model
import torch

augmentation_types = [
    # "Identity",
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
    # "AutoContrast",
    "Equalize",
]


def get_plot(augmentation_type, model, dataset_split=100):
    print(
        f"\n============================ Processing augmentation type: {augmentation_type} ============================\n"
    )
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

        # SSIM Calculation
        _, _, confidences = next(iter(trainloader))
        mean, std = get_mean_std(confidences)
        mean_list.append(mean.item())
        std_list.append(std.item())

        # Model Confidence Calculation
        if model is not None:
            evaluate_model(model=model, dataloader=trainloader)
        else:
            print(f'\nModel not provided. Skipping model evaluation.\n')

    plot_mean_std(mean_list, std_list, augmentation_type)

    print(
        f"\n============================ Finished: {augmentation_type} ============================\n"
    )

if __name__ == "__main__":
    # for augmentation_type in augmentation_types:
    #     get_plot(augmentation_type)
    get_plot("Brightness", model=None)

# print(f'Mean: {mean_list}\tStd: {std_list}')
