import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from CSV file
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
contrast_hvs = [0.32, 0.32, 0.64254054, 0.96603963, 0.96734732, 0.96865501, 0.9699627, 0.9712704, 0.97257809, 0.97388578, 0.97519347, 0.97650117, 0.97780886, 0.97911655, 0.98042424, 0.98173193, 0.98303963, 0.98434732, 0.98565501, 0.98696271, 0.9882704, 0.98957809, 0.99088578, 0.99219347, 0.99350117, 0.99480886, 0.99611655, 0.99742424, 0.99873194, 1., 1.]
contrast_hvs[31:] = 1 * np.ones(31, dtype=float)
df_contr = pd.read_csv(f'non_linear_mapping_data/Contrast/Contrast_poly_k_results.csv')
df_bright = pd.read_csv(f'non_linear_mapping_data/Brightness/Brightness_poly_k_results.csv')

df_contr = df_contr.sort_values(by='severity')
df_bright = df_bright.sort_values(by='severity')
contr_acc = np.array(df_contr['accuracy'])
bright_acc = np.array(df_bright['accuracy'])
severity = np.array(df_contr['severity'])


# Plotting
fig = plt.figure(figsize=(10, 8))
plt.plot(severity, contr_acc, label='Contrast Accuracy', color='green')
plt.plot(severity, bright_acc, label='Brightness Accuracy', color='red')
plt.plot(severity, contrast_hvs, label='Contrast HVS', linestyle='--', color='blue')
plt.fill_between(severity, contr_acc, bright_acc, color='gray', alpha=0.5, label='Difference')
plt.xlabel('Magnitude of Augmentation', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
file_name = '/home/ekagra/Documents/GitHub/MasterArbeit/final_plots/contrast_bright_acc.svg'
plt.savefig(file_name, format='svg')
plt.show()
