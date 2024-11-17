import numpy as np
import matplotlib.pyplot as plt

from curve_plotting import plot_severity_vs_confidence

ssim, ncc, scc, uiq, sift = plot_severity_vs_confidence("Rotate", "elapsed_time")

avg_ssim = np.mean(ssim) * 500
avg_ncc = np.mean(ncc) * 500
avg_scc = np.mean(scc) * 500
avg_uiq = np.mean(uiq) * 500
avg_sift = np.mean(sift) * 500

print(f"Average SSIM: {avg_ssim}")
print(f"Average NCC: {avg_ncc}")
print(f"Average SCC: {avg_scc}")
print(f"Average UIQ: {avg_uiq}")
print(f"Average SIFT: {avg_sift}")

plt.figure(figsize=(10, 5))
plt.bar(["SSIM", "NCC", "SCC", "UIQ", "SIFT"], [avg_ssim, avg_ncc, avg_scc, avg_uiq, avg_sift])
plt.xlabel("Metric")
plt.ylabel("Elapsed time per Image")
plt.title("Elapsed time per Image vs Metric")
plt.show()

