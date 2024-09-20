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


if __name__ == "__main__":
    """Occlusion"""
    # visibility_values = np.arange(0, 1, 0.1)
    # confidence_rc_values = get_data(visibility_values=visibility_values)

    # print(f'visibility_values: {visibility_values}')
    # print(f'confidence_rc_values: {confidence_rc_values}')

    # plot_data(visibility_values=visibility_values, confidence_rc_values=confidence_rc_values)

    """Rotation"""
    rotation_values1 = np.arange(0, 151, 30)
    confidence_values1 = [0.99, .99, .98, .97, .93, .96]

    rotation_values2 = np.arange(0, 181, 45)
    confidence_values2 = [0.98, .99, .94, .94, .88]

    k = 2
    fitted_values = []
    for i in range(len(rotation_values2)):
        rot_val = rotation_values2[i]
        visibility = rot_val / 360
        confidence_val = 1 - (1 - 0.1) * visibility ** k
        fitted_values.append(confidence_val)
    print(f'rotation_values: {rotation_values2}')
    print(f'fitted_values: {fitted_values}')
    print(f'confidence_values: {confidence_values2}')

    plt.figure(figsize=(10, 5))
    plt.plot(rotation_values2, fitted_values, label='Confidence vs Rotation')
    plt.plot(rotation_values2, confidence_values2, marker='o')
    plt.plot(rotation_values2, confidence_values2, label='Confidence vs Rotation')
    plt.show()
