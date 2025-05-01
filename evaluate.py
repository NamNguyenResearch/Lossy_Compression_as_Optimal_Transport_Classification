import os
import pandas as pd
import matplotlib.pyplot as plt

# def extract_last_epoch_results(base_dir):
#     """
#     Extracts the distortion loss and rate from the last epoch in each _perception_losses.csv file.
#     """
#     results = []
#     for folder in os.listdir(base_dir):
#         folder_path = os.path.join(base_dir, folder)
#         if os.path.isdir(folder_path):
#             csv_file = os.path.join(folder_path, "_perception_losses.csv")
#             if os.path.exists(csv_file):
#                 # Read the CSV file
#                 df = pd.read_csv(csv_file)
#                 # Get the last row (last epoch)
#                 last_row = df.iloc[-1]
#                 distortion_loss = last_row["distortion_loss"]
#                 rate = last_row["rate"]
#                 results.append((rate, distortion_loss))
#     return results

# def plot_distortion_loss_rate_curve(results, base_dir, output_filename="distortion_loss_rate_curve.png"):
#     """
#     Plots the distortion loss-rate curve and saves it to a file in the base_dir folder.
#     """
#     results = sorted(results, key=lambda x: x[0])  # Sort by rate
#     rates, distortion_losses = zip(*results)
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(rates, distortion_losses, marker='o', label="Distortion Loss vs Rate")
#     plt.xlabel("Rate")
#     plt.ylabel("Distortion Loss")
#     plt.title("Distortion Loss - Rate Curve")
#     plt.legend()
#     plt.grid(True)
    
#     # Save the figure in the base_dir folder
#     output_path = os.path.join(base_dir, output_filename)
#     plt.savefig(output_path)
#     plt.savefig(output_path) 

# if __name__ == "__main__":
#     base_dir = "experiments/M7/"  # Path to the directory containing folders like 4-4, 6-4, etc.
#     results = extract_last_epoch_results(base_dir)
#     if results:
#         plot_distortion_loss_rate_curve(results, base_dir)
#     else:
#         print("No valid _perception_losses.csv files found.")

def extract_last_epoch_results(base_dir):
    """
    Extracts the distortion loss and rate from the last epoch in each _perception_losses.csv file.
    """
    results = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            csv_file = os.path.join(folder_path, "_perception_losses.csv")
            if os.path.exists(csv_file):
                # Read the CSV file
                df = pd.read_csv(csv_file)
                # Get the last row (last epoch)
                last_row = df.iloc[-1]
                distortion_loss = last_row["distortion_loss"]
                rate = last_row["rate"]
                results.append((rate, distortion_loss))
    return results

def plot_distortion_loss_rate_curves(results_common_randomness, results_no_common_randomness, output_path="distortion_loss_rate_comparison.png"):
    """
    Plots two distortion loss-rate curves (for common randomness and no common randomness) and saves the figure to a file.
    """
    # Sort results by rate
    results_common_randomness = sorted(results_common_randomness, key=lambda x: x[0])
    results_no_common_randomness = sorted(results_no_common_randomness, key=lambda x: x[0])

    # Extract rates and distortion losses
    rates_common_randomness, distortion_losses_common_randomness = zip(*results_common_randomness)
    rates_no_common_randomness, distortion_losses_no_common_randomness = zip(*results_no_common_randomness)
    
    # Plot the curves
    plt.figure(figsize=(8, 6))
    plt.plot(rates_common_randomness, distortion_losses_common_randomness, marker='o', label="Common Randomness")
    plt.plot(rates_no_common_randomness, distortion_losses_no_common_randomness, marker='s', label="No Common Randomness")
    plt.xlabel("Rate")
    plt.ylabel("Distortion Loss")
    plt.title("Distortion Loss - Rate Curve Comparison")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    base_dir_common_randomness = "experiments/M3"  # Path to the directory containing folders for common randomness
    base_dir_no_common_randomness = "experiments/M4"  # Path to the directory containing folders for no common randomness

    # Extract results for common randomness and no common randomness
    results_common_randomness = extract_last_epoch_results(base_dir_common_randomness)
    results_no_common_randomness = extract_last_epoch_results(base_dir_no_common_randomness)

    if results_common_randomness and results_no_common_randomness:
        # Plot and save the comparison figure
        plot_distortion_loss_rate_curves(results_common_randomness, results_no_common_randomness, output_path="distortion_loss_rate_comparison.png")
    else:
        print("No valid _perception_losses.csv files found in one or both directories.")
