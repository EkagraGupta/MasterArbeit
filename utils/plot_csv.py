import pandas as pd
import matplotlib.pyplot as plt


def read_csv(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by="Severity")
    return df["Severity"], df["Mean"], df["Accuracy"]


def plot_mean_and_accuracy(file_paths, out_path, augmentation_type):
    data, accuracies = {}, {}
    # accuracy = None
    mean_acc = 0

    for k, file_path in enumerate(file_paths, start=1):
        severity, mean, acc = read_csv(file_path)
        data[k] = {"Severity": severity, "Mean": mean}
        accuracies[k] = {"Severity": severity, "Accuracy": acc}
        # if accuracy is None:
        #     accuracy = acc  # Store accuracy from the first file

    for k in range(1, len(data)):
        mean_acc += accuracies[k]["Accuracy"]
    mean_acc /= len(data) - 1

    # for k, values in data.items():
    #     plt.plot(values["Severity"], values["Mean"], label=f"k={k}")
    # plt.plot(data[1]["Severity"], data[1]["Mean"], label="Contrast (SSIM)")
    plt.plot(data[1]["Severity"], data[1]["Mean"], label="Luminance (SSIM)")
    plt.plot(data[2]["Severity"], data[2]["Mean"], label="NCC")
    plt.plot(data[3]["Severity"], data[3]["Mean"], label="SSIM")
    # plt.plot(data[3]["Severity"], data[3]["Mean"], label="SCC")
    plt.plot(data[4]["Severity"], data[4]["Mean"], label="UIQ")
    plt.plot(data[5]["Severity"], data[5]["Mean"], label="Sigmoid")
    # plt.plot(data[6]["Severity"], data[6]["Mean"], label="Custom Function")
    # plt.plot(data[7]["Severity"], data[7]["Mean"], label="VIF")

    plt.plot(severity, mean_acc, label="Model Accuracy", linestyle="--", color="black")

    plt.xlabel("Severity")
    plt.ylabel("Confidence")
    plt.title(
        f"Mean for Different Comparison Metrics of {augmentation_type} Augmentation"
    )
    plt.legend()
    plt.savefig(out_path)
    plt.show()


# Example usage
if __name__ == "__main__":
    augmentation_type = "Contrast"
    file_paths = [
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_contrast_ssim_results.csv",
        # f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_luminance_ssim_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_ncc_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_ssim_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_uiq_results.csv",
        # f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_scc_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_sigmoid_results.csv",
        # f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_test_results.csv",
        # f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_vif_results.csv",
    ]
    out_path = f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_results.png"
    plot_mean_and_accuracy(file_paths, out_path, augmentation_type)
