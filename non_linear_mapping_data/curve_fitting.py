import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def poly(x, a, b, c, d):
    result = a*x**3 + b*x**2 + c*x + d
    if np.any(result>1.0):
        result.where(result>1.0, 1.0, inplace=True)
    return result

def fit_poly(function, x, y):
    initial_guess = [1, 0, 1, np.mean(y)]
    popt, pcov = curve_fit(function, x, y, p0=initial_guess)
    return popt, pcov

def gaussian(x, a, b, c):
    gauss = a * np.exp(-0.5 * ((x - b) / c) ** 2)
    if np.any(gauss>1.0):
        gauss.where(gauss>1.0, 1.0, inplace=True)
    return gauss

def custom_function(x, a, b, c, d, e):
    # best values: [ 1.2438093   7.18937766 -0.87255438 -0.0573816  -0.2456411 ]
    result = a / (1.0 + np.exp(-b * (x - c))) + d * x + e
    if np.any(result>1.0):
        result.where(result>1.0, 1.0, inplace=True)
        
    return result

def fit_custom(function, x, y):
    initial_guess = [1, 0, 0, 0, 0]
    popt, pcov = curve_fit(function, x, y, p0=initial_guess)
    return popt, pcov

def fit_gaussian(function, x, y):
    popt, pcov = curve_fit(function, x, y, p0=[1, 0, 10])
    return popt, pcov

def sigmoid(x, a, b, c):
    return a / (1.0 + np.exp(-b * (x - c)))

def fit_sigmoid(function, x, y):
    popt, pcov = curve_fit(function, x, y, p0=[1, 1, 1])
    return popt, pcov

def plot_curves(function, augmentation_magnitude, model_accuracy, augmentation_mean, augmentation_std, popt):
    # Generate fitted values
    fitted_values = function(augmentation_magnitude, *popt)

    # Plot the original data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(augmentation_magnitude, model_accuracy,
            'ko--', label='Model Outputs')  # Original data
    plt.plot(augmentation_magnitude, fitted_values, 'b-',
            label='Fitted Gaussian Curve')  # Fitted curve
    plt.plot(augmentation_magnitude, augmentation_mean, 'r-',
            label='Confidence Mean')  # Fitted curve
    plt.fill_between(augmentation_magnitude, [m - s for m, s in zip(augmentation_mean, augmentation_std)],
                        [m + s for m, s in zip(augmentation_mean, augmentation_std)], color="red", alpha=0.2)
    plt.xlabel('Brightness Magnitude')
    plt.ylabel('Model Output')
    plt.title('Gaussian Fit to Model Output')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    augmentation_type = 'Brightness'
    data = pd.read_csv(
        f'/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_luminance_ssim_results.csv')
    data = data.sort_values(by='Severity')
    augmentation_magnitude = data['Severity']
    augmentation_mean = data['Mean']
    augmentation_std = data['Std']
    model_accuracy = data['Accuracy']

    # Fit the gaussian function to the data
    # popt, pcov = fit_gaussian(gaussian, augmentation_magnitude, model_accuracy)
    # print(f'Fitted Gaussian Parameters: {popt}')
    # plot_curves(augmentation_magnitude, model_accuracy, augmentation_mean, augmentation_std, popt)

    # Fit the two_poly function to the data
    popt, pcov = fit_sigmoid(sigmoid, augmentation_magnitude, model_accuracy)
    print(f'Fitted Two Poly Parameters: {popt}')
    plot_curves(sigmoid, augmentation_magnitude, model_accuracy, augmentation_mean, augmentation_std, popt)

    # fit the sigmoid function to the data
    # popt, pcov = fit_sigmoid(sigmoid, augmentation_magnitude, model_accuracy)
    # print(f'Fitted Sigmoid Parameters: {popt}')
    # plot_curves(augmentation_magnitude, model_accuracy, augmentation_mean, augmentation_std, popt)
