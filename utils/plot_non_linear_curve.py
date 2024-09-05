import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

COMPARISON_METRIC = "ssim"


def get_mean_std(confidences_tensor):
    mean = confidences_tensor.mean()
    std = confidences_tensor.std()

    return mean, std


def save_to_csv(
    mean_list,
    std_list,
    accuracy_list,
    augmentation_type,
    augmentation_magnitudes_list,
    time_list,
):
    folder_name = f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = os.path.join(
        folder_name, f"{augmentation_type}_{COMPARISON_METRIC}_results.csv"
    )

    # print(f'Augmentation Magnitudes: {augmentation_magnitudes}')

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Severity", "Mean", "Std", "Accuracy", "Time Taken"])
        for i, (mean, std, accuracy, augmentation_magnitude, time_taken) in enumerate(
            zip(
                mean_list,
                std_list,
                accuracy_list,
                augmentation_magnitudes_list,
                time_list,
            ),
            1,
        ):
            writer.writerow(
                [augmentation_magnitude.item(), mean, std, accuracy, time_taken]
            )
    print(f"Results saved to {filename}")
    return filename


def plot_mean_std(
    mean, std, model_confidences, augmentation_type=None, augmentation_magnitudes=[]
):
    folder_name = f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = os.path.join(
        folder_name, f"{augmentation_type}_{COMPARISON_METRIC}_results.png"
    )

    # x = range(1, len(mean) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(
        augmentation_magnitudes, mean, "-", color="red", label="Comparison Confidence"
    )
    # plt.errorbar(x, mean, yerr=std, fmt="o")
    plt.fill_between(
        augmentation_magnitudes,
        [m - s for m, s in zip(mean, std)],
        [m + s for m, s in zip(mean, std)],
        color="red",
        alpha=0.2,
    )
    plt.plot(
        augmentation_magnitudes,
        model_confidences,
        "-",
        color="blue",
        label="Model Confidence",
    )
    # plt.xticks(augmentation_magnitudes)
    plt.ylabel("Confidence")
    plt.title(f"Mean and standard deviation curve for {augmentation_type}")
    plt.legend()
    plt.savefig(filename)
    # plt.show()


def plot_mean_std_from_csv(csv_file, augmentation_type=None):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="Severity")

    # Extract the relevant columns
    mean = df["Mean"].tolist()
    std = df["Std"].tolist()
    model_confidences = df["Accuracy"].tolist()
    augmentation_magnitudes = df["Severity"].tolist()

    # if augmentation_type=='Solarize':
    #     augmentation_magnitudes.reverse()

    # Create folder if it doesn't exist
    folder_name = f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = os.path.join(
        folder_name, f"{augmentation_type}_{COMPARISON_METRIC}_results.png"
    )

    # Plot Mean and Std against Augmentation Magnitudes
    augmentation_magnitudes.sort()
    print(augmentation_magnitudes)
    plt.figure(figsize=(8, 6))
    plt.plot(
        augmentation_magnitudes, mean, "-", color="red", label="Comparison Confidence"
    )
    plt.fill_between(
        augmentation_magnitudes,
        [m - s for m, s in zip(mean, std)],
        [m + s for m, s in zip(mean, std)],
        color="red",
        alpha=0.2,
    )
    plt.plot(
        augmentation_magnitudes,
        model_confidences,
        "-",
        color="blue",
        label="Model Confidence",
    )
    plt.ylabel("Confidence")
    plt.title(f"Mean and standard deviation curve for {augmentation_type}")
    plt.legend()
    plt.savefig(filename)
    plt.show()


# Example usage:
if __name__ == "__main__":
    # mean = [0.5, 0.6, 0.9, 0.8, 0.7]
    # std = [1.0, 1.0, 0.3, 0.4, 0.5]
    # model_confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
    # augmentation_magnitudes = [1, 2, 3, 4, 5]
    # plot_mean_std(mean, std, model_confidences, "Brightness", augmentation_magnitudes)
    plot_mean_std_from_csv(
        "/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/Solarize/Solarize_ncc_results.csv",
        "Solarize",
    )
