import pandas as pd
from typing import Optional
import os


def model_accuracy_mapping(
    augmentation_magnitude: Optional[float], augmentation_type: Optional[str], root_path: Optional[str] = "/kaggle/working"
) -> Optional[float]:
    filename = os.path.join(root_path, f"{augmentation_type}_MAPPING_results.csv")
    data = pd.read_csv(filename)
    augmentation_magnitude_list = data["Severity"]
    model_accuracy_list = data["Accuracy"]

    # idx = np.where(augmentation_magnitude_list == augmentation_magnitude)
    for i in range(len(augmentation_magnitude_list)):
        mag = augmentation_magnitude_list[i]
        if round(mag, 5) == round(augmentation_magnitude, 5):
            return model_accuracy_list[i]


if __name__ == "__main__":

    augmentation_type = "Posterize"
    data = pd.read_csv(
        f"non_linear_mapping_data/{augmentation_type}/{augmentation_type}_MAPPING_results.csv"
    )

    augmentation_magnitude = data["Severity"]
    augmentation_mean = data["Mean"]
    augmentation_std = data["Std"]
    model_accuracy = data["Accuracy"]

    augmentation_value = augmentation_magnitude[25]
    print(augmentation_value)
    model_accuracy_value = model_accuracy_mapping(augmentation_value, augmentation_type)
    print(f"Model Accuracy: {model_accuracy_value}")
