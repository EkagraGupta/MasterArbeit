from augment_dataset import create_transforms, load_data
from utils.plot_non_linear_curve import (
    plot_mean_std,
    get_mean_std,
    save_to_csv,
    plot_mean_std_from_csv,
)
from wideresnet import WideResNet_28_4
from evaluate import evaluate_model
import torch
import time
from augment_dataset import seed_worker


def get_plot(augmentation_type, model, dataset_split=100, worker_init_fn=None, generator=None):
    print(
        f"\n============================ Processing augmentation type: {augmentation_type} ============================\n"
    )
    mean_list, std_list, time_list, augmentation_magnitudes_list = [], [], [], []
    accuracy_list = []

    for enable_sign in range(0, 2):
        if enable_sign == 1:
            enable_sign = False
        else:
            enable_sign = True

        for severity in range(0, 31):
            total_time = 0
            print(f"Processing severity: {severity} with sign: {enable_sign}\n")
            preprocess, augmentation = create_transforms(
                random_cropping=False,
                aggressive_augmentation=True,
                custom=True,
                augmentation_name=augmentation_type,
                augmentation_severity=severity,
                augmentation_sign=enable_sign,
                dataset_name='CIFAR10'
            )
            trainset, _ = load_data(
                transforms_preprocess=preprocess,
                transforms_augmentation=augmentation,
                dataset_split=dataset_split,
            )

            if isinstance(dataset_split, int):
                if worker_init_fn is not None:
                    dataloader = torch.utils.data.DataLoader(
                        trainset,
                        shuffle=True,
                        batch_size=dataset_split,
                        worker_init_fn=worker_init_fn,
                        generator=generator,
                    )
                else:
                    dataloader = torch.utils.data.DataLoader(
                        trainset,
                        shuffle=True,
                        batch_size=dataset_split,
                    )
            else:
                dataloader = torch.utils.data.DataLoader(
                    trainset,
                    shuffle=True,
                    batch_size=50000,
                )

            # Confidence Calculation
            start_time = time.time()
            _, _, confidences = next(iter(dataloader))
            augmentation_magnitude = confidences[0][0]
            confidences = confidences[1]
            end_time = time.time()
            mean, std = get_mean_std(confidences)
            mean_list.append(mean.item())
            std_list.append(std.item())
            augmentation_magnitudes_list.append(augmentation_magnitude)

            total_time += end_time - start_time
            print(f"Time taken: {total_time:.2f} seconds\n")
            time_list.append(total_time)

            # Model Confidence Calculation
            if model is not None:
                accuracy = evaluate_model(model=model, dataloader=dataloader)
                accuracy_list.append(accuracy)
                print(f"Accuracy: {accuracy*100:.2f}%")
            else:
                print(f"\nModel not provided. Skipping model evaluation.\n")

    # plot_mean_std(mean_list, std_list, accuracy_list,
    #               augmentation_type, augmentation_magnitudes_list)
    if len(accuracy_list) == 0:
        accuracy_list = [0] * len(mean_list)
    csv_filename = save_to_csv(
        mean_list,
        std_list,
        accuracy_list,
        augmentation_type,
        augmentation_magnitudes_list,
        time_list,
        iq_metric='sift'
    )
    print(f'CSV Filename: {csv_filename}')
    # plot_mean_std_from_csv(csv_file=csv_filename, augmentation_type=augmentation_type)

    print(
        f"\n============================ Finished: {augmentation_type} ============================\n"
    )


if __name__ == "__main__":
    augmentation_types = [
        # "Identity",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
        "Rotate",
        # "Brightness",
        # "Color",
        # "Contrast",
        # "Sharpness",
        # "Posterize",
        # "Solarize",
        # "AutoContrast",
        # "Equalize",
    ]

    g = torch.Generator()
    g.manual_seed(0)

    # Load the saved model weights
    net = WideResNet_28_4(num_classes=10)
    PATH = "/home/ekagra/Documents/GitHub/MasterArbeit/models/robust_no_TA_augments.pth"
    net = torch.nn.DataParallel(net)
    state_dict = torch.load(PATH, map_location=torch.device("cpu"))
    net.load_state_dict(state_dict["model_state_dict"], strict=False)

    for augmentation_type in augmentation_types:
        get_plot(augmentation_type, model=None, dataset_split=500, worker_init_fn=seed_worker, generator=g)
