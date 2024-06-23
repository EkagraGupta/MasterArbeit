def get_correction_factor(augmentation_info: dict) -> float:
    correction_factor = 0.
    for _, value in augmentation_info.items():
        correction_factor += float(value)
    return correction_factor