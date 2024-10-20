import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

COMPARISON_METRIC = "comparison_all"


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
    iq_metric
):
    folder_name = f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = os.path.join(
        folder_name, f"{augmentation_type}_{COMPARISON_METRIC}_results.csv"
    )

    print(f'mean: {mean_list}\nstd: {std_list}\naccuracy: {accuracy_list}\naugmentation_magnitudes: {augmentation_magnitudes_list}\ntime: {time_list}\n')

    if not os.path.exists(filename):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["severity", "accuracy", f"mean_{iq_metric}", f"std_{iq_metric}", f"elapsed_time_{iq_metric}"])
            for mean, std, accuracy, augmentation_magnitude, time_taken in zip(
                mean_list,
                std_list,
                accuracy_list,
                augmentation_magnitudes_list,
                time_list,
            ):
                writer.writerow(
                    [augmentation_magnitude.item(), accuracy, mean, std, time_taken]
                )
        print(f"Results saved to {filename}\n")
    else:
        df = pd.read_csv(filename)
        df[f'mean_{iq_metric}'] = mean_list
        df[f'std_{iq_metric}'] = std_list
        df[f'elapsed_time_{iq_metric}'] = time_list
        
        df.to_csv(filename, index=False)
        print(f'Results for Image Quality metric {iq_metric} appended to {filename}\n')

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


def plot_mean_std_from_csv(csv_file, augmentation_type=None, iq_metric='scc'):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="severity")

    # Extract the relevant columns
    mean = df[f"mean_{iq_metric}"].tolist()
    std = df[f"std_{iq_metric}"].tolist()
    model_confidences = df["accuracy"].tolist()
    augmentation_magnitudes = df["severity"].tolist()

    if augmentation_type == "Solarize":
        augmentation_magnitudes.reverse()

    # Create folder if it doesn't exist
    folder_name = f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = os.path.join(
        folder_name, f"{augmentation_type}_{COMPARISON_METRIC}_results.png"
    )

    # Plot Mean and Std against Augmentation Magnitudes
    augmentation_magnitudes.sort()
    plt.figure(figsize=(8, 6))
    plt.plot(
        augmentation_magnitudes, mean, "-", color="red", label="Comparison Confidence"
    )
    # plt.fill_between(
    #     augmentation_magnitudes,
    #     [m - s for m, s in zip(mean, std)],
    #     [m + s for m, s in zip(mean, std)],
    #     color="red",
    #     alpha=0.2,
    # )
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
    # plt.savefig(filename)
    plt.show()


# Example usage:
if __name__ == "__main__":
    # mean = [0.5, 0.6, 0.9, 0.8, 0.7]
    # std = [1.0, 1.0, 0.3, 0.4, 0.5]
    # model_confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
    # augmentation_magnitudes = [1, 2, 3, 4, 5]
    # plot_mean_std(mean, std, model_confidences, "Brightness", augmentation_magnitudes)

    augmentation_type = "Brightness"

    plot_mean_std_from_csv(
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_{COMPARISON_METRIC}_results.csv",
        f"{augmentation_type}",
    )
