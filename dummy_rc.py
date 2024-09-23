import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def get_data(visibility_values: list, k: int = 2, chance: float = 0.1):
    confidence_rc_values = []
    for i in range(len(visibility_values)):
        visibility = visibility_values[i]
        confidence_rc = 1 - (1 - chance) * (1 - visibility) ** k
        confidence_rc_values.append(confidence_rc)
    return confidence_rc_values


def plot_data(visibility_values: list, confidence_rc_values: list):
    plt.plot(visibility_values, confidence_rc_values)
    plt.xlabel('Visibility')
    plt.ylabel('Confidence')
    plt.title('Confidence vs Visibility')
    plt.show()

def rotation_func(rotation_angle, chance, k):
    visibility = rotation_angle / 360
    return 1 - (1 - chance) * visibility ** k

def contrast_func(contrast, k1, k2):
    chance = 0.1  # Fixed chance
    mean_contrast = np.mean(contrast)
    # return 1 - (1 - chance) * (1 / (1 + np.exp(-k * contrast)))
    return (1 - chance) * (1.0 / (1.0 + np.exp(-k1 * (contrast - k2))))


if __name__ == "__main__":
    """Occlusion"""
    # visibility_values = np.arange(0, 1, 0.1)
    # confidence_rc_values = get_data(visibility_values=visibility_values)

    # print(f'visibility_values: {visibility_values}')
    # print(f'confidence_rc_values: {confidence_rc_values}')

    # plot_data(visibility_values=visibility_values, confidence_rc_values=confidence_rc_values)

    """Rotation"""
    # rotation_values1 = np.arange(0, 151, 30)
    # confidence_values1 = [0.99, .99, .98, .97, .93, .96]

    # rotation_values2 = np.arange(0, 181, 45)
    # confidence_values2 = [0.98, .99, .94, .94, .88]

    # fitted_values = rotation_func(rotation_values2, 0.1, k=2.8)
    # print(fitted_values)
    # print(rotation_values2)

    # plt.figure(figsize=(10, 5))
    # plt.plot(rotation_values2, fitted_values, label='Confidence vs Rotation')
    # plt.plot(rotation_values2, confidence_values2, marker='o', label='Confidence vs Rotation')
    # plt.show()

    """Contrast"""
    contrast_values = np.array([1, 4, 6, 10, 100])
    confidence_values = [0., 0.1, 0.6, 0.95, 0.99]

    # plt.figure(figsize=(10, 5))
    # plt.plot(contrast_values, confidence_values, marker='o', label='Confidence vs Contrast')
    # plt.xlabel('Contrast %')
    # plt.ylabel('Recognition Confidence')
    # plt.show()
    
    popt, pcov = curve_fit(contrast_func, contrast_values, confidence_values)
    k1_opt = popt[0]
    k2_opt = popt[1]
    print(f'k_opt: {(k1_opt, k2_opt)}')

    fitted_values = contrast_func(contrast_values, k1_opt, k2_opt)
    print(f'fitted_values: {fitted_values}')

    plt.figure(figsize=(10, 5))
    plt.plot(contrast_values, confidence_values, marker='o', label='Confidence vs Contrast')
    plt.plot(contrast_values, fitted_values, label='Fitted Curve')
    plt.xlabel('Contrast %')
    plt.ylabel('Recognition Confidence')
    plt.show()
