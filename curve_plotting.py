import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_severity_vs_confidence(augmentation_type: str, data_cat: str = "mean"):
    
    filename = os.path.join(f"./non_linear_mapping_data/{augmentation_type}/{augmentation_type}_comparison_all_results.csv")
    df = pd.read_csv(filename)
    df = df.sort_values("severity")
    ssim_values = df[f"{data_cat}_ssim"]
    ncc_values = df[f"{data_cat}_ncc"]
    uiq_values = df[f"{data_cat}_uiq"]
    scc_values = df[f"{data_cat}_scc"]

    if augmentation_type in ["ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"]:
        sift_values = df[f"{data_cat}_sift"]
        return ssim_values, ncc_values, uiq_values, scc_values, sift_values
    return ssim_values, ncc_values, uiq_values, scc_values
    


if __name__ == "__main__":
    plot_severity_vs_confidence("Brightness")
