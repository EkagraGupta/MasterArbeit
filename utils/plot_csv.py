import pandas as pd
import matplotlib.pyplot as plt


def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df["Severity"], df["Mean"], df["Accuracy"]


def plot_mean_and_accuracy(file_paths, out_path):
    data, accuracies = {}, {}
    # accuracy = None
    mean_acc = 0

    for k, file_path in enumerate(file_paths, start=1):
        severity, mean, acc = read_csv(file_path)
        data[k] = {"Severity": severity, "Mean": mean}
        accuracies[k] = {'Severity': severity, 'Accuracy': acc}
        # if accuracy is None:
        #     accuracy = acc  # Store accuracy from the first file

    num_measure = 0
    for k in range(1, 4):
        num_measure += 1
        mean_acc += accuracies[k]['Accuracy']
    mean_acc /= num_measure

    # for k, values in data.items():
    #     plt.plot(values["Severity"], values["Mean"], label=f"k={k}")
    # plt.plot(data[1]["Severity"], data[1]["Mean"], label="psnr")
    # plt.plot(data[1]["Severity"], data[1]["Mean"], label="Luminance (SSIM)")
    plt.plot(data[1]["Severity"], data[1]["Mean"], label="NCC")
    plt.plot(data[2]["Severity"], data[2]["Mean"], label="SSIM")
    plt.plot(data[3]["Severity"], data[3]["Mean"], label="Custom Function")

    plt.plot(severity, mean_acc, label="Model Accuracy",
             linestyle="--", color="black")

    plt.xlabel("Severity")
    plt.ylabel("Value")
    plt.title("Mean for Different Comparison Metrics")
    plt.legend()
    plt.savefig(out_path)
    plt.show()


# Example usage
if __name__ == "__main__":
    augmentation_type = 'ShearY'
    file_paths = [
        # f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_luminance_ssim_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_ncc_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_ssim_results.csv",
        f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_test_results.csv"
    ]
    out_path = f"/home/ekagra/Documents/GitHub/MasterArbeit/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_results.png"
    plot_mean_and_accuracy(file_paths, out_path)
