import matplotlib.pyplot as plt
import numpy as np


# def get_data(visibility_values: list, k: int = 2, chance: float = 0.1):
#     confidence_rc_values = []
#     for i in range(len(visibility_values)):
#         visibility = visibility_values[i]
#         confidence_rc = 1 - (1 - chance) * (visibility) ** k
#         confidence_rc_values.append(confidence_rc)
#     confidence_rc_values = np.array(confidence_rc_values)
#     confidence_rc_values = np.clip(confidence_rc_values, chance, 1.0)
#     return confidence_rc_values


def get_data(visibility_values: list, k: int = 2, chance: float = 0.1):
    confidence_rc_values = []

    confidence_rc_values = 1 - (1 - chance) * (visibility_values) ** k
    # confidence_rc_values = 1 - (1 - chance) * (1 - visibility_values) ** k
    confidence_rc_values = np.clip(confidence_rc_values, chance, 1.0)

    return confidence_rc_values


def plot_data(
    visibility_values: list,
    confidence_rc_values: list,
    k: int = 2,
    augmentation_type: str = "occlusion",
):
    plt.plot(visibility_values, confidence_rc_values, marker="o", label=f"k={k}")
    plt.xlabel("Visibility")
    plt.ylabel("Confidence")
    plt.title(f"HVS for {augmentation_type}")
    plt.xticks(np.arange(visibility_values[0], visibility_values[-1] + 0.1, 15))
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    # """Occlusion"""
    # augmentation_type = 'occlusion'
    # k = 2
    # min_val, max_val = 0.0, 1.0
    # num_bins = 31
    # visibility_values1 = [0.0, .05, .10, .15, .20, .25, .30, .35, 1.0]
    # confidence_values1 = [0.22, 0.42, 0.44, 0.6, 0.56, 0.64, 0.62, 0.72, 1.0]
    # confidence_values2 = [0.18, 0.42, 0.62, 0.64, 0.62, 0.73, 0.8, 0.72, 1.0]
    # confidence_values3 = [0.22, 0.48, 0.62, 0.74, 0.72, 0.76, 0.78, 0.83, 1.0]
    # confidence_values4 = [0.24, 0.47, 0.6, 0.74, 0.72, 0.8, 0.86, 0.87, 1.0]
    # confidence_values5 = [0.22, 0.58, 0.64, 0.72, 0.78, 0.82, 0.76, 0.77, 1.0]
    # confidence_values = np.mean([confidence_values1, confidence_values2, confidence_values3, confidence_values4, confidence_values5], axis=0)

    # visibility_values_lim = np.linspace(min_val, max_val, num_bins)
    # confidence_values_lim = np.interp(visibility_values_lim, visibility_values1, confidence_values)
    # confidence_rc_values = get_data(visibility_values=visibility_values_lim, k=k)
    # # confidence_rc_values[0] = confidence_values_lim[0]
    # # print(min(confidence_rc_values))
    # # plot_data(visibility_values=visibility_values_lim, confidence_rc_values=confidence_values_lim, k=k)

    # # plt.plot(visibility_values1, confidence_values1, marker='o', label=f'25 ms', color='red')
    # # plt.plot(visibility_values1, confidence_values2, marker='o', label=f'50 ms', color='blue')
    # # plt.plot(visibility_values1, confidence_values3, marker='o', label=f'75 ms', color='green')
    # # plt.plot(visibility_values1, confidence_values4, marker='o', label=f'100 ms', color='yellow')
    # # plt.plot(visibility_values1, confidence_values5, marker='o', label=f'150 ms', color='purple')
    # # plt.xlabel("Visibility")
    # # plt.ylabel("Confidence")
    # # plt.title(f"HVS for {augmentation_type}")
    # # plt.legend()
    # # plt.show()

    # plt.plot(visibility_values_lim, confidence_values_lim, marker='o', label=f'Actual', color='red')
    # plt.plot(visibility_values_lim, confidence_rc_values, marker='o', label=f'Fitted', color='blue')
    # plt.xlabel("Visibility")
    # plt.ylabel("Confidence")
    # plt.yticks(np.arange(0.1, 1.1, 0.1))
    # plt.title(f"HVS for {augmentation_type}")
    # plt.legend()
    # plt.show()

    # """Rotate"""
    # augmentation_type = "Rotation"
    # k = 1
    # min_val, max_val = 0.0, 135.0
    # num_bins = 31

    # rotation_values1 = np.arange(0.0, 151.0, 30)
    # confidence_values1 = [0.99, 0.99, 0.98, 0.97, 0.93, 0.96]
    # confidence_values2 = [0.94, 0.96, 0.93, 0.91, 0.92, 0.86]
    # confidence_values3 = [0.97, 0.98, 0.97, 0.96, 0.96, 0.92]
    # confidence_values4 = [1.0, 1.0, 1.0, 0.98, 0.99, 0.98]
    # confidence_values = np.mean(
    #     [
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
    # confidence_values_lim = np.interp(
    #     rotation_values_lim, rotation_values, confidence_values
    # )
    # confidence_values_lim[0] = 1.0

    # chance = min(confidence_values_lim)

    # estimated_confidence_values = get_data(
    #     visibility_values=rotation_values_lim, k=k, chance=chance
    # )
    # estimated_confidence_values[0] = confidence_values_lim[0]
    # # estimated_confidence_values[-1] = confidence_values_lim[-1]

    # print(
    #     f"rotation_values: {rotation_values_lim}\nconfidence_values: {confidence_values_lim}\nEstimated: {estimated_confidence_values}"
    # )
    # # plot_data(visibility_values=rotation_values_lim, confidence_rc_values=confidence_values_lim, k=k, augmentation_type=augmentation_type)

    # plt.plot(
    #     rotation_values_lim,
    #     confidence_values_lim,
    #     marker="o",
    #     label=f"Actual",
    #     color="red",
    # )
    # # plt.plot(rotation_values_lim, estimated_confidence_values, marker='o', label=f'Fitted', color='blue')
    # print(confidence_values_lim)
    # plt.xlabel("Visibility")
    # plt.ylabel("Confidence")
    # plt.title(f"HVS for {augmentation_type}")
    # plt.gca().invert_xaxis()
    # plt.legend()
    # plt.show()

    """Contrast"""
    augmentation_type = 'Contrast'
    min_val, max_val = 0.0, 0.99
    k = 20
    num_bins = 31
    # contrast_values = np.linspace(min_val, max_val, num_bins)
    contrast_values1 = [0.04, 0.06, 0.1, 1.0]
    confidence_values1 = [0.0, 0.28, 0.96, 1.0]
    confidence_values2 = [0.06, 0.3, 0.97, 1.0]
    confidence_values3 = [0.2, 0.6, 0.94, 1.0]
    confidence_values4 = [0.58, 0.9, 0.98, 1.0]
    confidence_values5 = [0.76, 0.88, 0.98, 1.0]
    confidence_values = np.mean([confidence_values1, confidence_values2, confidence_values3, confidence_values4, confidence_values5], axis=0)

    contrast_values_lim = np.linspace(min_val, max_val, num_bins)
    confidence_values_lim = np.interp(contrast_values_lim, contrast_values1, confidence_values)
    chance = min(confidence_values_lim)
    print(f'chance: {chance}')
    estimated_confidence_values1 = get_data(visibility_values=contrast_values_lim, k=10, chance=chance)
    estimated_confidence_values2 = get_data(visibility_values=contrast_values_lim, k=20, chance=chance)
    estimated_confidence_values3 = get_data(visibility_values=contrast_values_lim, k=25, chance=chance)

    # plot_data(visibility_values=contrast_values_lim, confidence_rc_values=confidence_values_lim, augmentation_type=augmentation_type)

    plt.plot(contrast_values_lim, (confidence_values_lim), marker='o', label=f'Actual', color='red')
    plt.plot(contrast_values_lim, estimated_confidence_values1, marker='o', label=f'k=10', color='blue')
    plt.plot(contrast_values_lim, estimated_confidence_values2, marker='o', label=f'k=20', color='green')
    plt.plot(contrast_values_lim, estimated_confidence_values3, marker='o', label=f'k=25', color='black')
    plt.xlabel("Visibility")
    plt.ylabel("Confidence")
    plt.title(f"HVS for {augmentation_type}")
    plt.legend()
    plt.show()

    """Brightness"""
    """Color"""
    """Sharpness"""
    """ShearX"""
