from augment_dataset import create_transforms, load_data
from utils.plot_non_linear_curve import plot_mean_std, get_mean_std
from wideresnet import WideResNet_28_4
from evaluate import evaluate_model
import torch


def get_plot(augmentation_type, model, dataset_split=100):
    print(
        f"\n============================ Processing augmentation type: {augmentation_type} ============================\n"
    )
    mean_list, std_list = [], []
    accuracy_list = []

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
        dataloader = torch.utils.data.DataLoader(
            trainset,
            shuffle=False,
            batch_size=1,
        )

        # SSIM Calculation
        _, _, confidences = next(iter(dataloader))
        mean, std = get_mean_std(confidences)
        mean_list.append(mean.item())
        std_list.append(std.item())

        # Model Confidence Calculation
        if model is not None:
            accuracy = evaluate_model(model=model, dataloader=dataloader)
            accuracy_list.append(accuracy)
            # print(f'Accuracy: {accuracy:.2f}%')
        else:
            print(f"\nModel not provided. Skipping model evaluation.\n")

    plot_mean_std(mean_list, std_list, accuracy_list, augmentation_type)

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
        # "Color",
        # "Contrast",
        # "Sharpness",
        # "Posterize",
        # "Solarize",
        # "AutoContrast",
        # "Equalize",
    ]

    # Load the saved model weights
    net_path = "/home/ekagra/Desktop/Study/MA/code/models/cifar_net.pth"
    net = WideResNet_28_4(num_classes=10)
    net.load_state_dict(torch.load(net_path, map_location=torch.device("cpu")))
    # net.eval()

    for augmentation_type in augmentation_types:
        get_plot(augmentation_type, model=net, dataset_split=1000)   
