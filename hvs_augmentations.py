import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from augmentations.random_crop import RandomCrop


def model_confidence(augmentation_type: str):
    data = pd.read_csv(
        f"/home/ekagra/Documents/GitHub/MasterArbeit/{augmentation_type}_MAPPING_results.csv"
    )

    data = data.sort_values(by="Severity")
    data.reset_index(drop=True, inplace=True)
    augmentation_magnitude = data["Severity"]
    augmentation_mean = data["Mean"]
    augmentation_std = data["Std"]
    model_accuracy = data["Accuracy"]
    return augmentation_magnitude, augmentation_mean, model_accuracy


def get_data(visibility_values: list, k: int = 2, chance: float = 0.1):
    confidence_rc_values = []

    confidence_rc_values = 1 - (1 - chance) * (visibility_values) ** k
    confidence_rc_values = np.clip(confidence_rc_values, chance, 1.0)

    return confidence_rc_values


if __name__ == "__main__":
    """Occlusion"""
    # augmentation_type = 'occlusion'
    # min_val, max_val = 0.0, 1.0
    # num_bins = 31
    # visibility_values1 = [0.0, .05, .10, .15, .20, .25, .30, .35, 1.0]
    # confidence_values1 = [0.22, 0.42, 0.44, 0.6, 0.56, 0.64, 0.62, 0.72, 1.0]
    # confidence_values2 = [0.18, 0.42, 0.62, 0.64, 0.62, 0.73, 0.8, 0.72, 1.0]
    # confidence_values3 = [0.22, 0.48, 0.62, 0.74, 0.72, 0.76, 0.78, 0.83, 1.0]
    # confidence_values4 = [0.24, 0.47, 0.6, 0.74, 0.72, 0.8, 0.86, 0.87, 1.0]
    # confidence_values5 = [0.22, 0.58, 0.64, 0.72, 0.78, 0.82, 0.76, 0.77, 1.0]
    # confidence_values = np.mean([confidence_values1, confidence_values2, confidence_values3, confidence_values4, confidence_values5], axis=0)

    # k1, k2, k3 = 2, 3, 4

    # visibility_values_lim = np.linspace(min_val, max_val, num_bins)
    # visibility_values_lim_all = 2 * visibility_values_lim - 1
    # confidence_values_lim = np.interp(visibility_values_lim, visibility_values1, confidence_values)
    # chance = min(confidence_values_lim)
    # estimated_confidence_values1 = 1 - (1 - chance) * (1 - visibility_values_lim) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * (1 - visibility_values_lim) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * (1 - visibility_values_lim) ** k3

    # plt.plot(visibility_values_lim_all, confidence_values_lim, marker='o', label=f'Actual', color='red')
    # plt.plot(visibility_values_lim_all, estimated_confidence_values1, marker='o', label=f'k={k1}', color='blue')
    # # plt.plot(visibility_values_lim, estimated_confidence_values2, marker='o', label=f'k={k2}', color='green')
    # # plt.plot(visibility_values_lim, estimated_confidence_values3, marker='o', label=f'k={k3}', color='purple')
    # plt.xlabel("Visibility")
    # plt.ylabel("Confidence")
    # plt.yticks(np.arange(0.1, 1.1, 0.1))
    # plt.title(f"HVS for {augmentation_type}")
    # plt.legend()
    # plt.show()

    """Rotate"""
    # augmentation_type = "Rotate"
    # min_val, max_val = 0.0, 135.0
    # num_bins = 31

    # rotation_values1 = np.arange(0.0, 151.0, 30)
    # confidence_values1 = [1.0, 0.99, 0.98, 0.97, 0.93, 0.96]
    # confidence_values2 = [1.0, 0.96, 0.93, 0.91, 0.92, 0.86]
    # confidence_values3 = [1.0, 0.98, 0.97, 0.96, 0.96, 0.92]
    # confidence_values4 = [1.0, 1.0, 1.0, 0.98, 0.99, 0.98]
    # confidence_values = np.mean([
    #         confidence_values1,
    #         confidence_values2,
    #         confidence_values3,
    #         confidence_values4,
    #     ],
    #     axis=0,
    # )
    # rotation_values_another = np.arange(0.0, 181.0, 45)
    # confidence_values_another = [0.98, 0.99, 0.94, 0.94, 0.88]
    # rotation_values = np.concatenate((rotation_values1, rotation_values_another))
    # confidence_values = np.concatenate((confidence_values1, confidence_values_another))
    # unique_rot_vals, unique_indices = np.unique(rotation_values1, return_index=True)
    # rotation_values = unique_rot_vals.tolist()
    # confidence_values = confidence_values[unique_indices].tolist()

    # rotation_values_lim = np.linspace(min_val, max_val, num_bins)
    # confidence_values_lim = np.interp(rotation_values_lim, rotation_values, confidence_values)
    # confidence_values_lim[0] = 1.0

    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    # k1 = 1.5
    # k2 = 2
    # k3 = 3
    # chance = min(confidence_values_lim)
    # print(f"chance: {chance}")

    # estimated_confidence_values1 = 1 - (1 - chance) * (abs(augmentation_magnitude) / 135.0) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * (abs(augmentation_magnitude) / 135.0) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * (abs(augmentation_magnitude) / 135.0) ** k3
    # estimated_confidence_values1 = np.clip(estimated_confidence_values1, chance, 1.0)
    # estimated_confidence_values2 = np.clip(estimated_confidence_values2, chance, 1.0)
    # estimated_confidence_values3 = np.clip(estimated_confidence_values3, chance, 1.0)

    # plt.figure(figsize=(10, 6))
    # plt.plot(rotation_values_lim, confidence_values_lim, "--", label=f"Rotation HVS", color="red")
    # plt.plot(augmentation_magnitude, estimated_confidence_values1, '-', label=f'k={k1}', color='blue')
    # plt.plot(augmentation_magnitude, estimated_confidence_values2, '-', label=f'k={k2}', color='green')
    # plt.plot(augmentation_magnitude, estimated_confidence_values3, '-', label=f'k={k3}', color='purple')
    # # plt.plot(augmentation_magnitude, model_accuracy, "--", label="Model Confidence", color="black")
    # plt.xticks(rotation_values_lim[::10])
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()
    
    """Contrast"""
    
    # augmentation_type = 'Contrast'
    # num_bins = 31
    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    # contrast_values1 = [0.044, 0.061, 0.098, 0.956]
    # confidence_values1 = [0.0, 0.28, 0.96, 1.0]
    # confidence_values2 = [0.06, 0.3, 0.97, 1.0]
    # confidence_values3 = [0.2, 0.6, 0.94, 1.0]
    # confidence_values4 = [0.58, 0.9, 0.98, 1.0]
    # confidence_values5 = [0.76, 0.88, 0.98, 1.0]
    # confidence_values = np.mean([confidence_values1, confidence_values2,
    #                             confidence_values3, confidence_values4, confidence_values5], axis=0)
    # # contrast_values_extended = np.linspace(0.0, 1.0, num_bins)
    # contrast_values_extended = augmentation_magnitude[31:].copy()
    # confidence_values_interpolated = np.interp(contrast_values_extended, contrast_values1, confidence_values)
    # contrast_values_mapped = augmentation_magnitude[:31].copy()

    # k1 = 2
    # k2 = 10
    # k3 = 30
    # chance = min(confidence_values_interpolated)
    # print(f'{augmentation_type} chance: {chance}')

    # estimated_confidence_values1 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k3
    # estimated_confidence_values1 = np.clip(estimated_confidence_values1, chance, 1.0)
    # estimated_confidence_values2 = np.clip(estimated_confidence_values2, chance, 1.0)
    # estimated_confidence_values3 = np.clip(estimated_confidence_values3, chance, 1.0)

    # plt.figure(figsize=(10, 6))
    # plt.plot(contrast_values_mapped, confidence_values_interpolated, '--', label=f'Contrast HVS', color='red')
    # plt.plot(augmentation_magnitude, model_accuracy, "--", label="Model Confidence", color="black")
    # plt.plot(augmentation_magnitude, estimated_confidence_values1, '-', label=f'k={k1}', color='blue')
    # plt.plot(augmentation_magnitude, estimated_confidence_values2, '-', label=f'k={k2}', color='green')
    # plt.plot(augmentation_magnitude, estimated_confidence_values3, '-', label=f'k={k3}', color='purple')
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()

    """Brightness"""
    # augmentation_type = 'Brightness'
    # num_bins = 31
    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    # contrast_values1 = [0.044, 0.061, 0.098, 0.956]
    # confidence_values1 = [0.0, 0.28, 0.96, 1.0]
    # confidence_values2 = [0.06, 0.3, 0.97, 1.0]
    # confidence_values3 = [0.2, 0.6, 0.94, 1.0]
    # confidence_values4 = [0.58, 0.9, 0.98, 1.0]
    # confidence_values5 = [0.76, 0.88, 0.98, 1.0]
    # confidence_values = np.mean([confidence_values1, confidence_values2,
    #                             confidence_values3, confidence_values4, confidence_values5], axis=0)
    # # contrast_values_extended = np.linspace(0.0, 1.0, num_bins)
    # contrast_values_extended = augmentation_magnitude[31:].copy()
    # confidence_values_interpolated = np.interp(contrast_values_extended, contrast_values1, confidence_values)
    # contrast_values_mapped = augmentation_magnitude[:31].copy()

    # k1 = 2
    # k2 = 10
    # k3 = 30
    # chance = min(confidence_values_interpolated)
    # print(f'{augmentation_type} chance: {chance}')

    # estimated_confidence_values1 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k3
    # estimated_confidence_values1 = np.clip(estimated_confidence_values1, chance, 1.0)
    # estimated_confidence_values2 = np.clip(estimated_confidence_values2, chance, 1.0)
    # estimated_confidence_values3 = np.clip(estimated_confidence_values3, chance, 1.0)

    # plt.figure(figsize=(10, 6))
    # plt.plot(contrast_values_mapped, confidence_values_interpolated, '--', label=f'Contrast HVS', color='red')
    # plt.plot(augmentation_magnitude, model_accuracy, "--", label="Model Confidence", color="black")
    # plt.plot(augmentation_magnitude, estimated_confidence_values1, '-', label=f'k={k1}', color='blue')
    # plt.plot(augmentation_magnitude, estimated_confidence_values2, '-', label=f'k={k2}', color='green')
    # plt.plot(augmentation_magnitude, estimated_confidence_values3, '-', label=f'k={k3}', color='purple')
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()

    """Color"""
    # augmentation_type = 'Color'
    # num_bins = 31
    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    # contrast_values1 = [0.044, 0.061, 0.098, 0.956]
    # confidence_values1 = [0.0, 0.28, 0.96, 1.0]
    # confidence_values2 = [0.06, 0.3, 0.97, 1.0]
    # confidence_values3 = [0.2, 0.6, 0.94, 1.0]
    # confidence_values4 = [0.58, 0.9, 0.98, 1.0]
    # confidence_values5 = [0.76, 0.88, 0.98, 1.0]
    # confidence_values = np.mean([confidence_values1, confidence_values2,
    #                             confidence_values3, confidence_values4, confidence_values5], axis=0)
    # # contrast_values_extended = np.linspace(0.0, 1.0, num_bins)
    # contrast_values_extended = augmentation_magnitude[31:].copy()
    # confidence_values_interpolated = np.interp(contrast_values_extended, contrast_values1, confidence_values)
    # contrast_values_mapped = augmentation_magnitude[:31].copy()

    # k1 = 2
    # k2 = 10
    # k3 = 30
    # chance = min(confidence_values_interpolated)
    # print(f'{augmentation_type} chance: {chance}')

    # estimated_confidence_values1 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k3
    # estimated_confidence_values1 = np.clip(estimated_confidence_values1, chance, 1.0)
    # estimated_confidence_values2 = np.clip(estimated_confidence_values2, chance, 1.0)
    # estimated_confidence_values3 = np.clip(estimated_confidence_values3, chance, 1.0)

    # plt.figure(figsize=(10, 6))
    # plt.plot(contrast_values_mapped, confidence_values_interpolated, '--', label=f'Contrast HVS', color='red')
    # plt.plot(augmentation_magnitude, model_accuracy, "--", label="Model Confidence", color="black")
    # plt.plot(augmentation_magnitude, estimated_confidence_values1, '-', label=f'k={k1}', color='blue')
    # plt.plot(augmentation_magnitude, estimated_confidence_values2, '-', label=f'k={k2}', color='green')
    # plt.plot(augmentation_magnitude, estimated_confidence_values3, '-', label=f'k={k3}', color='purple')
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()

    # """Sharpness"""
    # augmentation_type = 'Sharpness'
    # num_bins = 31
    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    # contrast_values1 = [0.044, 0.061, 0.098, 0.956]
    # confidence_values1 = [0.0, 0.28, 0.96, 1.0]
    # confidence_values2 = [0.06, 0.3, 0.97, 1.0]
    # confidence_values3 = [0.2, 0.6, 0.94, 1.0]
    # confidence_values4 = [0.58, 0.9, 0.98, 1.0]
    # confidence_values5 = [0.76, 0.88, 0.98, 1.0]
    # confidence_values = np.mean([confidence_values1, confidence_values2,
    #                             confidence_values3, confidence_values4, confidence_values5], axis=0)
    # # contrast_values_extended = np.linspace(0.0, 1.0, num_bins)
    # contrast_values_extended = augmentation_magnitude[31:].copy()
    # confidence_values_interpolated = np.interp(contrast_values_extended, contrast_values1, confidence_values)
    # contrast_values_mapped = augmentation_magnitude[:31].copy()

    # k1 = 2
    # k2 = 10
    # k3 = 30
    # chance = min(confidence_values_interpolated)
    # print(f'{augmentation_type} chance: {chance}')

    # estimated_confidence_values1 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k3
    # estimated_confidence_values1 = np.clip(estimated_confidence_values1, chance, 1.0)
    # estimated_confidence_values2 = np.clip(estimated_confidence_values2, chance, 1.0)
    # estimated_confidence_values3 = np.clip(estimated_confidence_values3, chance, 1.0)

    # plt.figure(figsize=(10, 6))
    # plt.plot(contrast_values_mapped, confidence_values_interpolated, '--', label=f'Contrast HVS', color='red')
    # plt.plot(augmentation_magnitude, model_accuracy, "--", label="Model Confidence", color="black")
    # plt.plot(augmentation_magnitude, estimated_confidence_values1, '-', label=f'k={k1}', color='blue')
    # plt.plot(augmentation_magnitude, estimated_confidence_values2, '-', label=f'k={k2}', color='green')
    # plt.plot(augmentation_magnitude, estimated_confidence_values3, '-', label=f'k={k3}', color='purple')
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()


    """ShearX"""
    # augmentation_type = "ShearX"
    # min_val, max_val = 0.0, 135.0
    # num_bins = 31

    # rotation_values1 = np.arange(0.0, 151.0, 30)
    # confidence_values1 = [1.0, 0.99, 0.98, 0.97, 0.93, 0.96]
    # confidence_values2 = [1.0, 0.96, 0.93, 0.91, 0.92, 0.86]
    # confidence_values3 = [1.0, 0.98, 0.97, 0.96, 0.96, 0.92]
    # confidence_values4 = [1.0, 1.0, 1.0, 0.98, 0.99, 0.98]
    # confidence_values = np.mean([
    #         confidence_values1,
    #         confidence_values2,
    #         confidence_values3,
    #         confidence_values4,
    #     ],
    #     axis=0,
    # )
    # rotation_values_another = np.arange(0.0, 181.0, 45)
    # confidence_values_another = [0.98, 0.99, 0.94, 0.94, 0.88]
    # # merge the two lists
    # rotation_values = np.concatenate((rotation_values1, rotation_values_another))
    # confidence_values = np.concatenate((confidence_values1, confidence_values_another))
    # unique_rot_vals, unique_indices = np.unique(rotation_values1, return_index=True)
    # rotation_values = unique_rot_vals.tolist()
    # confidence_values = confidence_values[unique_indices].tolist()

    # rotation_values_lim = np.linspace(min_val, max_val, num_bins)
    # confidence_values_lim = np.interp(rotation_values_lim, rotation_values, confidence_values)
    # confidence_values_lim[0] = 1.0

    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    # chance = min(confidence_values_lim)
    # print(f"chance: {chance}")

    # k1 = 1.5
    # k2 = 2
    # k3 = 3

    # estimated_confidence_values1 = 1 - (1 - chance) * abs(augmentation_magnitude) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * abs(augmentation_magnitude) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * abs(augmentation_magnitude) ** k3
    # estimated_confidence_values1 = np.clip(estimated_confidence_values1, chance, 1.0)
    # estimated_confidence_values2 = np.clip(estimated_confidence_values2, chance, 1.0)
    # estimated_confidence_values3 = np.clip(estimated_confidence_values3, chance, 1.0)

    # plt.figure(figsize=(10, 6))
    # plt.plot(rotation_values_lim / 135., confidence_values_lim, "--", label=f"Rotation HVS", color="red")
    # plt.plot(augmentation_magnitude, estimated_confidence_values1, '-', label=f'k={k1}', color='blue')
    # plt.plot(augmentation_magnitude, estimated_confidence_values2, '-', label=f'k={k2}', color='green')
    # plt.plot(augmentation_magnitude, estimated_confidence_values3, '-', label=f'k={k3}', color='purple')
    # # plt.plot(augmentation_magnitude, model_accuracy, "--", label="Model Confidence", color="black")
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()

    """ShearY"""
    # augmentation_type = "ShearY"
    # min_val, max_val = 0.0, 135.0
    # num_bins = 31

    # rotation_values1 = np.arange(0.0, 151.0, 30)
    # confidence_values1 = [1.0, 0.99, 0.98, 0.97, 0.93, 0.96]
    # confidence_values2 = [1.0, 0.96, 0.93, 0.91, 0.92, 0.86]
    # confidence_values3 = [1.0, 0.98, 0.97, 0.96, 0.96, 0.92]
    # confidence_values4 = [1.0, 1.0, 1.0, 0.98, 0.99, 0.98]
    # confidence_values = np.mean([
    #         confidence_values1,
    #         confidence_values2,
    #         confidence_values3,
    #         confidence_values4,
    #     ],
    #     axis=0,
    # )
    # rotation_values_another = np.arange(0.0, 181.0, 45)
    # confidence_values_another = [0.98, 0.99, 0.94, 0.94, 0.88]
    # # merge the two lists
    # rotation_values = np.concatenate((rotation_values1, rotation_values_another))
    # confidence_values = np.concatenate((confidence_values1, confidence_values_another))
    # unique_rot_vals, unique_indices = np.unique(rotation_values1, return_index=True)
    # rotation_values = unique_rot_vals.tolist()
    # confidence_values = confidence_values[unique_indices].tolist()

    # rotation_values_lim = np.linspace(min_val, max_val, num_bins)
    # confidence_values_lim = np.interp(rotation_values_lim, rotation_values, confidence_values)
    # confidence_values_lim[0] = 1.0

    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    # chance = min(confidence_values_lim)
    # print(f"chance: {chance}")

    # k1 = 1.5
    # k2 = 2
    # k3 = 3

    # estimated_confidence_values1 = 1 - (1 - chance) * abs(augmentation_magnitude) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * abs(augmentation_magnitude) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * abs(augmentation_magnitude) ** k3
    # estimated_confidence_values1 = np.clip(estimated_confidence_values1, chance, 1.0)
    # estimated_confidence_values2 = np.clip(estimated_confidence_values2, chance, 1.0)
    # estimated_confidence_values3 = np.clip(estimated_confidence_values3, chance, 1.0)

    # plt.figure(figsize=(10, 6))
    # plt.plot(rotation_values_lim / 135., confidence_values_lim, "--", label=f"Rotation HVS", color="red")
    # plt.plot(augmentation_magnitude, estimated_confidence_values1, '-', label=f'k={k1}', color='blue')
    # plt.plot(augmentation_magnitude, estimated_confidence_values2, '-', label=f'k={k2}', color='green')
    # plt.plot(augmentation_magnitude, estimated_confidence_values3, '-', label=f'k={k3}', color='purple')
    # # plt.plot(augmentation_magnitude, model_accuracy, "--", label="Model Confidence", color="black")
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()

    """TranslateX"""
    # augmentation_type = "TranslateX"
    # min_val, max_val = 0.0, 1.0
    # num_bins = 31
    # visibility_values1 = [0.0, .05, .10, .15, .20, .25, .30, .35, 1.0]
    # confidence_values1 = [0.22, 0.42, 0.44, 0.6, 0.56, 0.64, 0.62, 0.72, 1.0]
    # confidence_values2 = [0.18, 0.42, 0.62, 0.64, 0.62, 0.73, 0.8, 0.72, 1.0]
    # confidence_values3 = [0.22, 0.48, 0.62, 0.74, 0.72, 0.76, 0.78, 0.83, 1.0]
    # confidence_values4 = [0.24, 0.47, 0.6, 0.74, 0.72, 0.8, 0.86, 0.87, 1.0]
    # confidence_values5 = [0.22, 0.58, 0.64, 0.72, 0.78, 0.82, 0.76, 0.77, 1.0]
    # confidence_values = np.mean([confidence_values1, confidence_values2, confidence_values3, confidence_values4, confidence_values5], axis=0)

    # translate_values_lim = np.linspace(min_val, max_val, num_bins)[::-1]
    # confidence_values_lim = np.interp(translate_values_lim, visibility_values1, confidence_values)[::-1]

    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)
    # random_crop = RandomCrop(n_class=10)
    # visibility = random_crop.compute_visibility(dim1=32, dim2=32, tx=augmentation_magnitude, ty=0.)

    # k1, k2, k3 = 2, 3, 4
    # chance = min(confidence_values_lim)
    # print(f'{augmentation_type} chance: {chance}')

    # # estimated_confidence_values1 = 1 - (1 - chance) * (abs(augmentation_magnitude) / 32.0) ** k1
    # # estimated_confidence_values2 = 1 - (1 - chance) * (abs(augmentation_magnitude) / 32.0) ** k2
    # # estimated_confidence_values3 = 1 - (1 - chance) * (abs(augmentation_magnitude) / 32.0) ** k3
    # estimated_confidence_values1 = 1 - (1 - chance) * (1 - visibility) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * (1 - visibility) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * (1 - visibility) ** k3

    # plt.figure(figsize=(10, 6))
    # plt.plot(translate_values_lim, confidence_values_lim, '--', label=f'Occlussion HVS', color='black')
    # plt.plot(augmentation_magnitude / 32.0, model_accuracy, '--', label=f'Model Accuracy', color='red')
    # plt.plot(augmentation_magnitude / 32.0, estimated_confidence_values1, '-', label=f'k={k1}', color='blue')
    # plt.plot(augmentation_magnitude / 32.0, estimated_confidence_values2, '-', label=f'k={k2}', color='green')
    # plt.plot(augmentation_magnitude / 32.0, estimated_confidence_values3, '-', label=f'k={k3}', color='purple')
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.yticks(np.arange(0.1, 1.1, 0.1))
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()

    """TranslateY"""
    # augmentation_type = "TranslateY"
    # min_val, max_val = 0.0, 1.0
    # num_bins = 31
    # visibility_values1 = [0.0, .05, .10, .15, .20, .25, .30, .35, 1.0]
    # confidence_values1 = [0.22, 0.42, 0.44, 0.6, 0.56, 0.64, 0.62, 0.72, 1.0]
    # confidence_values2 = [0.18, 0.42, 0.62, 0.64, 0.62, 0.73, 0.8, 0.72, 1.0]
    # confidence_values3 = [0.22, 0.48, 0.62, 0.74, 0.72, 0.76, 0.78, 0.83, 1.0]
    # confidence_values4 = [0.24, 0.47, 0.6, 0.74, 0.72, 0.8, 0.86, 0.87, 1.0]
    # confidence_values5 = [0.22, 0.58, 0.64, 0.72, 0.78, 0.82, 0.76, 0.77, 1.0]
    # confidence_values = np.mean([confidence_values1, confidence_values2, confidence_values3, confidence_values4, confidence_values5], axis=0)

    # translate_values_lim = np.linspace(min_val, max_val, num_bins)[::-1]
    # confidence_values_lim = np.interp(translate_values_lim, visibility_values1, confidence_values)[::-1]

    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)
    # random_crop = RandomCrop(n_class=10)
    # visibility = random_crop.compute_visibility(dim1=32, dim2=32, tx=0., ty=augmentation_magnitude)

    # k1, k2, k3 = 2, 3, 4
    # chance = min(confidence_values_lim)
    # print(f'{augmentation_type} chance: {chance}')

    # # estimated_confidence_values1 = 1 - (1 - chance) * (abs(augmentation_magnitude) / 32.0) ** k1
    # # estimated_confidence_values2 = 1 - (1 - chance) * (abs(augmentation_magnitude) / 32.0) ** k2
    # # estimated_confidence_values3 = 1 - (1 - chance) * (abs(augmentation_magnitude) / 32.0) ** k3
    # estimated_confidence_values1 = 1 - (1 - chance) * (1 - visibility) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * (1 - visibility) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * (1 - visibility) ** k3

    # plt.figure(figsize=(10, 6))
    # plt.plot(translate_values_lim, confidence_values_lim, '--', label=f'Occlussion HVS', color='black')
    # plt.plot(augmentation_magnitude / 32.0, model_accuracy, '--', label=f'Model Accuracy', color='red')
    # plt.plot(augmentation_magnitude / 32.0, estimated_confidence_values1, '-', label=f'k={k1}', color='blue')
    # plt.plot(augmentation_magnitude / 32.0, estimated_confidence_values2, '-', label=f'k={k2}', color='green')
    # plt.plot(augmentation_magnitude / 32.0, estimated_confidence_values3, '-', label=f'k={k3}', color='purple')
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.yticks(np.arange(0.1, 1.1, 0.1))
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()


    """Posterize"""
    # augmentation_type = 'Posterize'

    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    # unique_augmentation_magnitudes, unique_indices = np.unique(augmentation_magnitude, return_index=True)
    # unique_model_accuracy = model_accuracy[unique_indices]

    # k1 = 2
    # k2 = 3
    # k3 = 4
    # k4 = 5
    # chance = min(model_accuracy)
    # print(f"Minimum Chance: {chance}")

    # estimated_confidence_scores1 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes / 8.0) ** k1
    # estimated_confidence_scores2 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes / 8.0) ** k2
    # estimated_confidence_scores3 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes / 8.0) ** k3
    # estimated_confidence_scores4 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes / 8.0) ** k4
    
    # # # plot the curves
    # plt.figure(figsize=(10, 6))
    # plt.plot(unique_augmentation_magnitudes, unique_model_accuracy, "--", label="Model Outputs", color="black")
    # plt.plot(unique_augmentation_magnitudes, estimated_confidence_scores1, "-", label=f"k={k1}", color="blue")
    # plt.plot(unique_augmentation_magnitudes, estimated_confidence_scores2, "-", label=f"k={k2}", color="green")
    # plt.plot(unique_augmentation_magnitudes, estimated_confidence_scores3, "-", label=f"k={k3}", color="purple")
    # plt.plot(unique_augmentation_magnitudes, estimated_confidence_scores4, "-", label=f"k={k4}", color="magenta")
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()


    """Solarize"""
    augmentation_type = 'Solarize'

    augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    unique_augmentation_magnitudes, unique_indices = np.unique(augmentation_magnitude, return_index=True)
    unique_model_accuracy = model_accuracy[unique_indices]

    k1 = 1.5
    k2 = 1
    k3 = 2
    k4 = 3
    chance = min(model_accuracy)
    print(f"Minimum Chance: {chance}")

    estimated_confidence_scores1 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes / 255.0) ** k1
    estimated_confidence_scores2 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes / 255.0) ** k2
    estimated_confidence_scores3 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes / 255.0) ** k3
    estimated_confidence_scores4 = 1 - (1 - chance) * (1 - unique_augmentation_magnitudes / 255.0) ** k4
    
    # # plot the curves
    plt.figure(figsize=(10, 6))
    plt.plot(unique_augmentation_magnitudes, unique_model_accuracy, "--", label="Model Outputs", color="black")
    plt.plot(unique_augmentation_magnitudes, estimated_confidence_scores1, "-", label=f"k={k1}", color="blue")
    plt.plot(unique_augmentation_magnitudes, estimated_confidence_scores2, "-", label=f"k={k2}", color="green")
    plt.plot(unique_augmentation_magnitudes, estimated_confidence_scores3, "-", label=f"k={k3}", color="purple")
    plt.plot(unique_augmentation_magnitudes, estimated_confidence_scores4, "-", label=f"k={k4}", color="magenta")
    plt.xlabel(f"Magnitude of {augmentation_type}")
    plt.ylabel("Confidence")
    plt.title(f"{augmentation_type} with chance: {chance}")
    plt.legend()
    plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    plt.show()


    """DUMMY"""
    # augmentation_type = 'Contrast'
    # num_bins = 31
    # augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    # contrast_values1 = [0.044, 0.061, 0.098, 0.956]
    # confidence_values1 = [0.0, 0.28, 0.96, 1.0]
    # confidence_values2 = [0.06, 0.3, 0.97, 1.0]
    # confidence_values3 = [0.2, 0.6, 0.94, 1.0]
    # confidence_values4 = [0.58, 0.9, 0.98, 1.0]
    # confidence_values5 = [0.76, 0.88, 0.98, 1.0]
    # confidence_values = np.mean([confidence_values1, confidence_values2,
    #                             confidence_values3, confidence_values4, confidence_values5], axis=0)
    # # contrast_values_extended = np.linspace(0.0, 1.0, num_bins)
    # contrast_values_extended = augmentation_magnitude[31:].copy()
    # confidence_values_interpolated = np.interp(contrast_values_extended, contrast_values1, confidence_values)
    # contrast_values_mapped = augmentation_magnitude[:31].copy()

    # k1 = 2
    # k2 = 10
    # k3 = 30
    # chance = min(confidence_values_interpolated)
    # print(f'{augmentation_type} chance: {chance}')

    # estimated_confidence_values1 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k1
    # estimated_confidence_values2 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k2
    # estimated_confidence_values3 = 1 - (1 - chance) * (abs(augmentation_magnitude)) ** k3
    # estimated_confidence_values1 = np.clip(estimated_confidence_values1, chance, 1.0)
    # estimated_confidence_values2 = np.clip(estimated_confidence_values2, chance, 1.0)
    # estimated_confidence_values3 = np.clip(estimated_confidence_values3, chance, 1.0)

    # plt.figure(figsize=(10, 6))
    # plt.plot(contrast_values_mapped, confidence_values_interpolated, '--', label=f'Contrast HVS', color='red')
    # plt.plot(augmentation_magnitude, model_accuracy, "--", label="Model Confidence", color="black")
    # plt.plot(augmentation_magnitude, estimated_confidence_values1, '-', label=f'k={k1}', color='blue')
    # plt.plot(augmentation_magnitude, estimated_confidence_values2, '-', label=f'k={k2}', color='green')
    # plt.plot(augmentation_magnitude, estimated_confidence_values3, '-', label=f'k={k3}', color='purple')
    # plt.xlabel(f"Magnitude of {augmentation_type}")
    # plt.ylabel("Confidence")
    # plt.title(f"{augmentation_type} with chance: {chance}")
    # plt.legend()
    # plt.savefig(f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_plot.png")
    # plt.show()