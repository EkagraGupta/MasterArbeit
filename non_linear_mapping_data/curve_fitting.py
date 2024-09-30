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
    # for i in range(len(visibility_values)):
    #     visibility = visibility_values[i]
    #     confidence_rc = 1 - (1 - chance) * (1 - visibility) ** k
    #     confidence_rc_values.append(confidence_rc)
    #     print(f"Visibility: {visibility}, Confidence: {confidence_rc}")

    confidence_rc_values = 1 - (1 - chance) * (1 - np.array(visibility_values)) ** k
    confidence_rc_values = np.array(confidence_rc_values)
    confidence_rc_values = np.where(confidence_rc_values < chance, chance, confidence_rc_values)
    confidence_rc_values = np.where(confidence_rc_values > 1.0, 1.0, confidence_rc_values)
    return confidence_rc_values

def compute_visibility(dim1: int, dim2: int, tx: float, ty: float) -> float:
    """Computes the visibility of the cropped uimage within the background.

    Args:
        dim1 (int): Height of the image.
        dim2 (int): Width of the image.
        tx (int): Horizontal offset.
        ty (int): Vertical offset.

    Returns:
        float: Visibility ratio of the cropped image.
    """
    return (dim1 - abs(tx)) * (dim2 - abs(ty)) / (dim1 * dim2)

if __name__ == "__main__":
    augmentation_type = "TranslateX"
    data = pd.read_csv(
        f"non_linear_mapping_data/{augmentation_type}/{augmentation_type}_MAPPING_results.csv"
    )
    data = data.sort_values(by="Severity")
    augmentation_magnitude = data["Severity"]
    augmentation_mean = data["Mean"]
    augmentation_std = data["Std"]
    model_accuracy = data["Accuracy"]

    # visibility = compute_visibility(
    #     dim1=32.0, dim2=32.0, tx=0.0, ty=augmentation_magnitude
    # )
    confidence_rc_values1 = get_nl_curve(augmentation_magnitude, k=2.0, chance=min(model_accuracy))
    confidence_rc_values2 = get_nl_curve(augmentation_magnitude, k=3.0, chance=min(model_accuracy))
    confidence_rc_values3 = get_nl_curve(augmentation_magnitude, k=5.0, chance=min(model_accuracy))
    confidence_rc_values4 = get_nl_curve(augmentation_magnitude, k=10.0, chance=min(model_accuracy))
    confidence_rc_values5 = get_nl_curve(augmentation_magnitude, k=15.0, chance=min(model_accuracy))
    confidence_rc_values6 = get_nl_curve(augmentation_magnitude, k=18.0, chance=min(model_accuracy))
    confidence_rc_values7 = get_nl_curve(augmentation_magnitude, k=20.0, chance=min(model_accuracy))

    # plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(augmentation_magnitude, model_accuracy, "ko--", label="Model Outputs", color="red")
    plt.plot(augmentation_magnitude, confidence_rc_values1, "b-", label=f"k={2}", color="blue")
    plt.plot(augmentation_magnitude, confidence_rc_values2, "b-", label=f"k={3}", color="green")
    plt.plot(augmentation_magnitude, confidence_rc_values3, "b-", label=f"k={5}", color="orange")
    plt.plot(augmentation_magnitude, confidence_rc_values4, "b-", label=f"k={10}", color="purple")
    plt.plot(augmentation_magnitude, confidence_rc_values5, "b-", label=f"k={15}", color="black")
    plt.plot(augmentation_magnitude, confidence_rc_values6, "b-", label=f"k={18}", color="brown")
    plt.plot(augmentation_magnitude, confidence_rc_values7, "b-", label=f"k={20}", color="olive")
    # plt.plot(augmentation_magnitude, visibility, "b-", label="Fitted Curve")
    plt.xlabel("Severity")
    plt.ylabel("Confidence")
    plt.title(f"Non Linear Mapping for {augmentation_type}")
    plt.legend()
    plt.savefig(f"non_linear_mapping_data/{augmentation_type}/{augmentation_type}_MAPPING_results.png")
    plt.show()


    # Fit the gaussian function to the data
    # popt, pcov = fit_poly_3(poly_3, augmentation_magnitude, model_accuracy)
    # print(f"Fitted Gaussian Parameters: {popt}")
    # plot_curves(
    #     gaussian,
    #     augmentation_magnitude,
    #     model_accuracy,
    #     augmentation_mean,
    #     augmentation_std,
    #     popt,
    # )

    # severity = np.array([0, 15, 30])
    # visibility = np.array([1.0, 0.8, 0.5])

    # popt, pcov = fit_poly_2(poly_2, severity, visibility)
    # print(f"Fitted Parameters: {popt}")

    # y_fit = poly_2(15, *popt)
    # print(y_fit)

    # print(f'Augmentation Magnitude: {augmentation_magnitude}\nModel Accuracy: {model_accuracy}')
