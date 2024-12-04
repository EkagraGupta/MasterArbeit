import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from CSV file
augmentation_type = 'Solarize'
df_model = pd.read_csv(f'non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
df_aug = pd.read_csv(f'non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')

df_model = df_model.sort_values(by='severity')
df_aug = df_aug.sort_values(by='severity')

model_acc = np.array(df_model['accuracy'])
aug_acc = np.array(df_aug['mean_poly_k'])
severity = np.array(df_model['severity'])

print(len(model_acc), len(aug_acc), len(severity))

# Plotting
fig = plt.figure(figsize=(10, 5))
plt.plot(severity, model_acc, label='Model Accuracy', marker='o')
plt.plot(severity, aug_acc, label='Augmentation Accuracy', marker='o')
plt.show()

