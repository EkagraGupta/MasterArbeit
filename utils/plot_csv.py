import pandas as pd
import matplotlib.pyplot as plt


def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df["Severity"], df["Mean"], df["Accuracy"]


def plot_mean_and_accuracy(file_paths):
    data = {}
    accuracy = None

    for k, file_path in enumerate(file_paths, start=1):
        severity, mean, acc = read_csv(file_path)
        data[k] = {"Severity": severity, "Mean": mean}
        if accuracy is None:
            accuracy = acc  # Store accuracy from the first file

    # for k, values in data.items():
    #     plt.plot(values["Severity"], values["Mean"], label=f"k={k}")
    # plt.plot(data[1]["Severity"], data[1]["Mean"], label="psnr")
    plt.plot(data[1]["Severity"], data[1]["Mean"], label="scc")
    plt.plot(data[2]["Severity"], data[2]["Mean"], label="lightning ssim")
    plt.plot(data[3]["Severity"], data[3]["Mean"], label="uiq")
    plt.plot(data[4]["Severity"], data[4]["Mean"], label="ncc")
    plt.plot(data[5]["Severity"], data[5]["Mean"], label="ssim")

    plt.plot(severity, accuracy, label="Model Accuracy",
             linestyle="--", color="black")

    plt.xlabel("Severity")
    plt.ylabel("Value")
    plt.title("Mean for Different Comparison Metrics")
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    augmentation_type = 'Solarize'
    file_paths = [
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_l_scc_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_l_ssim_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_l_uiq_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_ncc_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_ssim_results.csv"
    ]
    plot_mean_and_accuracy(file_paths)
