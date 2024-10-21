import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_severity_vs_confidence(augmentation_type: str):
    
    filename = os.path.join(f"./non_linear_mapping_data/{augmentation_type}/{augmentation_type}_comparison_all_results.csv")
    df = pd.read_csv(filename)
    df = df.sort_values("severity")
    ssim_values = df["mean_ssim"]
    ncc_values = df["mean_ncc"]
    uiq_values = df["mean_uiq"]
    scc_values = df["mean_scc"]
    return ssim_values, ncc_values, uiq_values, scc_values
    


if __name__ == "__main__":
    plot_severity_vs_confidence("Brightness")
