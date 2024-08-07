import matplotlib.pyplot as plt


def get_mean_std(confidences_tensor):
    mean = confidences_tensor.mean()
    std = confidences_tensor.std()

    return mean, std


def plot_mean_std(mean, std, model_confidences, augmentation_type=None):
    x = range(1, len(mean) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, mean, "-", color="red", label="SSIM Confidence")
    # plt.errorbar(x, mean, yerr=std, fmt="o")
    plt.fill_between(x, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color="red", alpha=0.2)
    plt.plot(x, model_confidences, "-", color="blue", label="Model Confidence")
    plt.xticks([1], ["Severity"])
    plt.ylabel("Value")
    plt.title(f"Mean and standard deviation curve for {augmentation_type}")
    plt.legend()
    plt.savefig(
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}.png"
    )
    plt.show()


# Example usage:
if __name__ == "__main__":
    mean = [0.5, 0.6, 0.9, 0.8, 0.7]
    std = [0.1, 0.2, 0.3, 0.4, 0.5]
    model_confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
    plot_mean_std(mean, std, model_confidences, "Brightness")
