def get_correction_factor(augmentation_info: dict) -> float:
    correction_factor = 0.0
    threshold = 0.1
    for _, value in augmentation_info.items():
        correction_factor += float(value)
        print(correction_factor)
    return abs(correction_factor) if 1.>abs(correction_factor)>threshold else 0.5
