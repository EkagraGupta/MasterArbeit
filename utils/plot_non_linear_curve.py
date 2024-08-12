import matplotlib.pyplot as plt
import os
import csv


def get_mean_std(confidences_tensor):
    mean = confidences_tensor.mean()
    std = confidences_tensor.std()

    return mean, std

def save_to_csv(mean_list, std_list, accuracy_list, augmentation_type):
    folder_name = f'/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = os.path.join(folder_name, f'{augmentation_type}_results.csv')

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Severity", "Mean", "Std", "Accuracy"])
        for i, (mean, std, accuracy) in enumerate(
            zip(mean_list, std_list, accuracy_list), 1
        ):
            writer.writerow([i, mean, std, accuracy])
    print(f"Results saved to {filename}")

def plot_mean_std(mean, std, model_confidences, augmentation_type=None):
    folder_name = f'/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = os.path.join(folder_name, f'{augmentation_type}_results.png')

    x = range(1, len(mean) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, mean, "-", color="red", label="Comparison Confidence")
    # plt.errorbar(x, mean, yerr=std, fmt="o")
    plt.fill_between(x, [m - s for m, s in zip(mean, std)],
                     [m + s for m, s in zip(mean, std)], color="red", alpha=0.2)
    plt.plot(x, model_confidences, "-", color="blue", label="Model Confidence")
    plt.xticks([1], ["Severity"])
    plt.ylabel("Value")
    plt.title(f"Mean and standard deviation curve for {augmentation_type}")
    plt.legend()
    plt.savefig(filename)
    # plt.show()


# Example usage:
if __name__ == "__main__":
    mean = [0.5, 0.6, 0.9, 0.8, 0.7]
    std = [0.1, 0.2, 0.3, 0.4, 0.5]
    model_confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
    plot_mean_std(mean, std, model_confidences, "Brightness")
