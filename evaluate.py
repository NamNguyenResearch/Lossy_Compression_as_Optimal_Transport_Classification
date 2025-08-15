import os
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================================
def extract_last_epoch_results(base_dir, loss_file="_perception_losses.csv", accuracy_file="_accuracy.csv"):
    """
    Extracts the distortion loss, rate, and accuracy from the last epoch of the given CSV files.
    """
    loss_results = []
    accuracy_results = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            # Extract distortion loss and rate
            loss_csv = os.path.join(folder_path, loss_file)
            if os.path.exists(loss_csv):
                df_loss = pd.read_csv(loss_csv)
                last_row = df_loss.iloc[-1]
                distortion_loss = last_row["distortion_loss"]
                rate = last_row["rate"]
                loss_results.append((rate, distortion_loss))

            # Extract accuracy and rate
            acc_csv = os.path.join(folder_path, accuracy_file)
            if os.path.exists(acc_csv):
                df_acc = pd.read_csv(acc_csv)
                last_row = df_acc.iloc[-1]
                accuracy = last_row["accuracy"]
                rate = last_row["rate"]
                accuracy_results.append((rate, accuracy))

    return loss_results, accuracy_results

# =============================================================================================
def plot_curves(loss_results_cr, loss_results_no_cr, accuracy_results_cr, accuracy_results_no_cr, output_path_prefix="comparison"):
    """
    Plots distortion loss-rate and accuracy-rate curves, saving both figures.
    """
    # Sort the results
    loss_results_cr = sorted(loss_results_cr, key=lambda x: x[0])
    loss_results_no_cr = sorted(loss_results_no_cr, key=lambda x: x[0])
    accuracy_results_cr = sorted(accuracy_results_cr, key=lambda x: x[0])
    accuracy_results_no_cr = sorted(accuracy_results_no_cr, key=lambda x: x[0])

    # Distortion loss vs Rate
    rates_cr, distortion_cr = zip(*loss_results_cr)
    rates_no_cr, distortion_no_cr = zip(*loss_results_no_cr)

    plt.figure(figsize=(8, 6))
    plt.plot(rates_cr, distortion_cr, marker='o', label="Common Randomness")
    # plt.plot(rates_no_cr, distortion_no_cr, marker='s', label="No Common Randomness")
    plt.xlabel("Rate")
    plt.ylabel("Distortion Loss")
    plt.title("Distortion Loss vs Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path_prefix}_distortion_loss_rate.png")
    print(f"Distortion loss figure saved to {output_path_prefix}_distortion_loss_rate.png")

    # Accuracy vs Rate
    rates_cr, accuracy_cr = zip(*accuracy_results_cr)
    rates_no_cr, accuracy_no_cr = zip(*accuracy_results_no_cr)

    plt.figure(figsize=(8, 6))
    plt.plot(rates_cr, accuracy_cr, marker='o', label="Common Randomness")
    # plt.plot(rates_no_cr, accuracy_no_cr, marker='s', label="No Common Randomness")
    plt.xlabel("Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path_prefix}_accuracy_rate.png")
    print(f"Accuracy figure saved to {output_path_prefix}_accuracy_rate.png")

# =============================================================================================
if __name__ == "__main__":
    base_dir_cr = "experiments/S1"  # Common randomness
    base_dir_no_cr = "experiments/S2"  # No common randomness

    loss_cr, acc_cr = extract_last_epoch_results(base_dir_cr)
    loss_no_cr, acc_no_cr = extract_last_epoch_results(base_dir_no_cr)

    if loss_cr and loss_no_cr and acc_cr and acc_no_cr:
        plot_curves(loss_cr, loss_no_cr, acc_cr, acc_no_cr)
    else:
        print("Missing data in one or more directories.")