import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

augmentation_type = 'Rotate'
data = pd.read_csv(
    f'/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_ssim_results.csv')
data = data.sort_values(by='Severity')
augmentation_magnitude = data['Severity']
augmentation_mean = data['Mean']
augmentation_std = data['Std']
model_accuracy = data['Accuracy']


def gaussian(x, a, b, c):
    gauss = a * np.exp(-0.5 * ((x - b) / c) ** 2)
    if np.any(gauss>1.0):
        gauss.where(gauss>1.0, 1.0, inplace=True)
    return gauss


# fit the gaussian function to the data
popt, pcov = curve_fit(gaussian, augmentation_magnitude,
                       model_accuracy, p0=[1, 0, 10])

# Extract the parameters of fitted gaussian
a, b, c = popt

# Generate fitted values
fitted_values = gaussian(augmentation_magnitude, *popt)

print(f'Fitted Gaussian Parameters: a={a:.2f}, b={b:.2f}, c={c:.2f}')

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
