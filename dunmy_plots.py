import matplotlib.pyplot as plt
import numpy as np
from hvs_augmentations import model_confidence
import seaborn as sns
import scienceplots
from curve_plotting import plot_severity_vs_confidence

plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
plt.style.use(['science', 'no-latex'])
main_data_color = '#377eb8'
secondary_data_color1 = '#4daf4a'
secondary_data_color2 = '#ff7f00'
highlight_color = '#e41a1c'
est_conf_color = 'red'
metrics_color = '#333333'

augmentation_type1 = "Brightness"
augmentation_type2 = 'Contrast'
contrast_hvs = [0.32, 0.32, 0.64254054, 0.96603963, 0.96734732, 0.96865501, 0.9699627, 0.9712704, 0.97257809, 0.97388578, 0.97519347, 0.97650117, 0.97780886, 0.97911655, 0.98042424, 0.98173193, 0.98303963, 0.98434732, 0.98565501, 0.98696271, 0.9882704, 0.98957809, 0.99088578, 0.99219347, 0.99350117, 0.99480886, 0.99611655, 0.99742424, 0.99873194, 1., 1.]
    
num_bins = 31
augmentation_magnitude, _, model_accuracy = model_confidence(augmentation_type=augmentation_type2)
augmentation_magnitude1, _, model_accuracy1 = model_confidence(augmentation_type=augmentation_type1)
contrast_hvs = contrast_hvs + [1.0] * 31

plt.figure(figsize=(12, 8))
plt.plot(augmentation_magnitude, contrast_hvs, '--', label='Contrast HVS', color=main_data_color, linewidth=2.5, alpha=0.8)
plt.plot(augmentation_magnitude, model_accuracy, '-', label="Model Confidence (Contrast)", color=secondary_data_color1, linewidth=3)
plt.plot(augmentation_magnitude, model_accuracy1, '-', label="Model Confidence (Brightness)", color=secondary_data_color2, linewidth=3)
plt.fill_between(augmentation_magnitude, model_accuracy, model_accuracy1, color=highlight_color, alpha=0.2, label="Difference Highlight")
plt.xlabel("Magnitude of Augmentations", fontsize=14, labelpad=10)
plt.ylabel("Confidence", fontsize=14, labelpad=10)
plt.legend(fontsize=12, frameon=False, loc='lower right', edgecolor=metrics_color)
file_name = f"/home/ekagra/Documents/GitHub/MasterArbeit/final_plots/contrast_brightness_plot.png"
plt.savefig(file_name)
plt.tight_layout()
plt.show()

