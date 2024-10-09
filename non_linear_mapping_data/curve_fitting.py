import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def poly(x, a, b, c, d):
    result = a * x**3 + b * x**2 + c * x + d
    if np.any(result > 1.0):
        result.where(result > 1.0, 1.0, inplace=True)
    return result


def fit_poly(function, x, y):
    initial_guess = [1, 0, 1, np.mean(y)]
    popt, pcov = curve_fit(function, x, y, p0=initial_guess)
    return popt, pcov


def custom_function(x, a, b, c, d, e):
    # best values: [ 1.2438093   7.18937766 -0.87255438 -0.0573816  -0.2456411 ]
    result = a / (1.0 + np.exp(-b * (x - c))) + d * x + e
    if np.any(result > 1.0):
        result.where(result > 1.0, 1.0, inplace=True)

    return result


def fit_custom(function, x, y):
    initial_guess = [1, 0, 0, 0, 0]
    popt, pcov = curve_fit(function, x, y, p0=initial_guess)
    return popt, pcov


def gaussian(x, a, b, c, d):
    gauss = d + a * np.exp(-0.5 * ((x - b) / c) ** 2)
    if np.any(gauss > 1.0):
        gauss.where(gauss > 1.0, 1.0, inplace=True)
    return gauss


def fit_gaussian(function, x, y):
    popt, pcov = curve_fit(function, x, y, p0=[5, 0, 10, 2])
    return popt, pcov


def sigmoid(x, a, b, c):
    return a / (1.0 + np.exp(-b * (x - c)))


def fit_sigmoid(function, x, y):
    popt, pcov = curve_fit(function, x, y, p0=[1, 1, 1])
    return popt, pcov


def poly_2(x, a, b):
    y = a * x**2 + b
    y = np.where(y > 1.0, 1.0, y)
    return y


def fit_poly_2(function, x, y):
    initial_guess = [1.0, 1.0]
    popt, pcov = curve_fit(function, x, y, p0=initial_guess)
    return popt, pcov


def plot_curves(
    function,
    augmentation_magnitude,
    model_accuracy,
    augmentation_mean,
    augmentation_std,
    popt,
):
    # Generate fitted values
    fitted_values = function(augmentation_magnitude, *popt)

    # Plot the original data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(
        augmentation_magnitude, model_accuracy, "ko--", label="Model Outputs"
    )  # Original data
    plt.plot(
        augmentation_magnitude, fitted_values, "b-", label="Fitted Gaussian Curve"
    )  # Fitted curve
    plt.plot(
        augmentation_magnitude, augmentation_mean, "r-", label="Confidence Mean"
    )  # Fitted curve
    plt.fill_between(
        augmentation_magnitude,
        [m - s for m, s in zip(augmentation_mean, augmentation_std)],
        [m + s for m, s in zip(augmentation_mean, augmentation_std)],
        color="red",
        alpha=0.2,
    )
    plt.xlabel("Brightness Magnitude")
    plt.ylabel("Model Output")
    plt.title("Gaussian Fit to Model Output")
    plt.legend()
    plt.show()


def model_accuracy_mapping(
    augmentation_magnitude: Optional[float], augmentation_type: Optional[str]
) -> Optional[float]:
    data = pd.read_csv(
        f"non_linear_mapping_data/{augmentation_type}/{augmentation_type}_MAPPING_results.csv"
    )
    augmentation_magnitude_list = data["Severity"]
    model_accuracy_list = data["Accuracy"]

    # idx = np.where(augmentation_magnitude_list == augmentation_magnitude)
    for i in range(len(augmentation_magnitude_list)):
        mag = augmentation_magnitude_list[i]
        if round(mag, 5) == round(augmentation_magnitude, 5):
            return model_accuracy_list[i]


def get_nl_curve(visibility_values: list, k: int = 2, chance: float = 0.1):
    confidence_rc_values = []

    confidence_rc_values = 1 - (1 - chance) * (visibility_values) ** k
    # confidence_rc_values = 1 - (1 - chance) * (1 - visibility_values) ** k
    confidence_rc_values = np.clip(confidence_rc_values, chance, 1.0)

    return confidence_rc_values


def compute_visibility(dim1: int, dim2: int, t: float) -> float:
    """Computes the visibility of the cropped uimage within the background.

    Args:
        dim1 (int): Height of the image.
        dim2 (int): Width of the image.
        tx (int): Horizontal offset.
        ty (int): Vertical offset.

    Returns:
        float: Visibility ratio of the cropped image.
    """
    return (dim1 - abs(t)) * dim2 / (dim1 * dim2)


if __name__ == "__main__":
    augmentation_type = "Posterize"
    data = pd.read_csv(
        f"/home/ekagra/Documents/GitHub/MasterArbeit/{augmentation_type}_MAPPING_results.csv"
    )

    data = data.sort_values(by="Severity")
    data.reset_index(drop=True, inplace=True)
    augmentation_magnitude = data["Severity"]
    augmentation_mean = data["Mean"]
    augmentation_std = data["Std"]
    model_accuracy = data["Accuracy"]

    # visibility = compute_visibility(
    #     dim1=32.0, dim2=32.0, t=augmentation_magnitude
    # )
    # visibility = augmentation_magnitude / 255.0
    # visibility_abs = abs(augmentation_magnitude)

    """TEST Posterize and Solarize"""
    augmentation_magnitudes = augmentation_magnitude
    # augmentation_magnitudes = augmentation_magnitude
    unique_augmentation_magnitudes, unique_indices = np.unique(augmentation_magnitudes, return_index=True)
    unique_model_accuracy = model_accuracy[unique_indices]

    k1 = 1
    k2 = 2
    k3 = 3
    k4 = 4
    chance = min(model_accuracy)
    print(f"Minimum Chance: {chance}")
    confidence_scores1 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes) ** k1
    confidence_scores2 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes) ** k2
    confidence_scores3 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes) ** k3
    confidence_scores4 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes) ** k4
    
    # # plot the curves
    plt.figure(figsize=(10, 6))
    plt.plot(unique_augmentation_magnitudes, unique_model_accuracy, "--", label="Model Outputs", color="red")
    plt.plot(unique_augmentation_magnitudes, confidence_scores1, "-", label=f"k={k1}", color="blue")
    # plt.plot(unique_augmentation_magnitudes, confidence_scores2, "-", label=f"k={k2}", color="green")
    # plt.plot(unique_augmentation_magnitudes, confidence_scores3, "-", label=f"k={k3}", color="purple")
    # plt.plot(unique_augmentation_magnitudes, confidence_scores4, "-", label=f"k={k4}", color="magenta")
    plt.legend()
    plt.show()
    """TEST"""

    # chance = min(model_accuracy)
    # print(f"Minimum Chance: {chance}")
    # k1, k2, k3, k4 = 2, 4, 6, 10

    # augmentation_magnitude_pos = (augmentation_magnitude + 1.0) / 2.0

    # confidence_value1 = 1 - (1 - chance) * (1 - augmentation_magnitude_pos) ** k1
    # confidence_value2 = 1 - (1 - chance) * (1 - augmentation_magnitude_pos) ** k2
    # confidence_value3 = 1 - (1 - chance) * (1 - augmentation_magnitude_pos) ** k3
    # confidence_value4 = 1 - (1 - chance) * (1 - augmentation_magnitude_pos) ** k4
    # # confidence_rc_values1 = get_nl_curve(visibility, k=k1, chance=chance)
    # # confidence_rc_values2 = get_nl_curve(visibility_abs, k=k2, chance=chance)
    # # confidence_rc_values1[augmentation_magnitude>0.0] = 1.0
    # # confidence_rc_values2[augmentation_magnitude>0.0] = 1.0

    # # idx = 5
    # # print(
    # #     f"Augmentation Magnitude: {augmentation_magnitude[idx]}\tModel Accuracy: {model_accuracy[idx]}\tConfidence RC Values: {confidence_rc_values1[idx]}"
    # # )

    # # # plot the curves
    # plt.figure(figsize=(10, 6))
    # plt.plot(
    #     augmentation_magnitude, model_accuracy, "--", label="Model Outputs", color="red"
    # )
    # plt.plot(
    #     augmentation_magnitude, augmentation_mean, "--", label="Augmentation Mean", color="pink"
    # )
    # plt.plot(
    #     augmentation_magnitude, augmentation_mean, "-", label="Augmentation mean", color="black"
    # )
    # plt.plot(
    #     augmentation_magnitude,
    #     confidence_value1,
    #     "-",
    #     label=f"k={k1}",
    #     color="blue",
    # )
    # plt.plot(
    #     augmentation_magnitude,
    #     confidence_value2,
    #     "-",
    #     label=f"k={k2}",
    #     color="green",
    # )
    # plt.plot(
    #     augmentation_magnitude,
    #     confidence_value3,
    #     "-",
    #     label=f"k={k3}",
    #     color="purple",
    # )
    # plt.plot(
    #     augmentation_magnitude,
    #     confidence_value4,
    #     "-",
    #     label=f"k={k4}",
    #     color="pink",
    # )
    # plt.legend()
    # plt.show()
