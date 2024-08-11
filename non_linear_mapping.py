from augment_dataset import create_transforms, load_data
from utils.plot_non_linear_curve import plot_mean_std, get_mean_std, save_to_csv
from wideresnet import WideResNet_28_4
# import wrn as wideresnet
from evaluate import evaluate_model
import torch
import csv
import time
import os



def get_plot(augmentation_type, model, dataset_split=100):
    print(
        f"\n============================ Processing augmentation type: {augmentation_type} ============================\n"
    )
    mean_list, std_list, time_list = [], [], []
    accuracy_list = []

    for severity in range(0, 30):
        total_time = 0
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

        if isinstance(dataset_split, int):
            dataloader = torch.utils.data.DataLoader(
                trainset,
                shuffle=False,
                batch_size=dataset_split,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                trainset,
                shuffle=False,
                batch_size=10000,
            )

        # SSIM Calculation
        start_time = time.time()
        _, _, confidences = next(iter(dataloader))
        end_time = time.time()
        mean, std = get_mean_std(confidences)
        mean_list.append(mean.item())
        std_list.append(std.item())

        total_time += end_time - start_time
        print(f"Time taken: {total_time:.2f} seconds\n")
        time_list.append(total_time)
        # Model Confidence Calculation
        if model is not None:
            accuracy = evaluate_model(model=model, dataloader=dataloader)
            accuracy_list.append(accuracy)
            print(f'Accuracy: {accuracy*100:.2f}%')
        else:
            print(f"\nModel not provided. Skipping model evaluation.\n")

    plot_mean_std(mean_list, std_list, accuracy_list, augmentation_type)
    save_to_csv(mean_list, std_list, accuracy_list, augmentation_type)

    print(
        f"\n============================ Finished: {augmentation_type} ============================\n"
    )


if __name__ == "__main__":
    augmentation_types = [
        # "Identity",
        # "ShearX",
        # "ShearY",
        # "TranslateX",
        # "TranslateY",
        # "Rotate",
        "Brightness",
        "Color",
        "Contrast",
        "Sharpness",
        "Posterize",
        "Solarize",
        # "AutoContrast",
        # "Equalize",
    ]

    # Load the saved model weights
    net = WideResNet_28_4(num_classes=10)
    PATH = '/home/ekagra/Documents/GitHub/MasterArbeit/models/robust_no_TA_augments.pth'
    net = torch.nn.DataParallel(net)
    state_dict = torch.load(PATH, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict["model_state_dict"], strict=False)

    for augmentation_type in augmentation_types:
        get_plot(augmentation_type, model=net, dataset_split=500)
