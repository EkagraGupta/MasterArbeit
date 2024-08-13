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
    plt.plot(data[1]["Severity"], data[1]["Mean"], label="psnr")
    plt.plot(data[2]["Severity"], data[2]["Mean"], label="scc")
    plt.plot(data[3]["Severity"], data[3]["Mean"], label="ssim")
    plt.plot(data[4]["Severity"], data[4]["Mean"], label="uiq")
    plt.plot(data[5]["Severity"], data[5]["Mean"], label="ncc")

    plt.plot(severity, accuracy, label="Model Accuracy", linestyle="--", color="black")

    plt.xlabel("Severity")
    plt.ylabel("Value")
    plt.title("Mean and Accuracy Curves for Different k Values")
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # file_paths = [
    #     '/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/TranslateX_k1_results.csv',
    #     '/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/TranslateX_k2_results.csv',
    #     '/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/TranslateX_k3_results.csv',
    #     '/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/TranslateX_k4_results.csv'
    # ]
    file_paths = [
        "/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/ShearX/ShearX_l_psnr_results.csv",
        "/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/ShearX/ShearX_l_scc_results.csv",
        "/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/ShearX/ShearX_l_ssim_results.csv",
        "/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/ShearX/ShearX_l_uiq_results.csv",
        "/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/ShearX/ShearX_ncc_results.csv"
    ]
    plot_mean_and_accuracy(file_paths)
