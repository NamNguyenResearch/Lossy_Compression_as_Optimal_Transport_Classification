import os
import subprocess
import datetime
import time
import json
import shutil


def run(experiment_path_pre, settings, commands):
    with open(f"{experiment_path_pre}/_status.tmp", "w") as f:
        now = datetime.datetime.now()
        f.write(now.strftime(f"Experiment: %d/%m/%Y %H:%M:%S\n"))
        f.write(
            "exp,"
            + ",".join(f"{key}" for key, value in settings[0].items())
            + f",timing\n"
        )

    for i, (setting, command) in enumerate(zip(settings, commands)):
        print(f"--- Beginning experiment {i} ---")
        start_time = time.process_time()
        subprocess.call(
            command, shell=True
        )  # The experiment will save results on its own
        end_time = time.process_time()
        minutes = (end_time - start_time) / 60

        with open(f"{experiment_path_pre}/_status.tmp", "a+") as f:
            f.write(
                f"{i},"
                + ",".join(f"{value}" for key, value in setting.items())
                + f",{minutes}\n"
            )


def train_with_params(mode, settings, experiment_number, selection_method=None, submode=None, overwrite=False):
    commands = []
    experiment_path = ""
    experiment_path_pre = f"experiments/{experiment_number}"

    for setting in settings:
        experiment_path_super_res = f"{setting['latent_dim']}-{setting['L']}"

        experiment_path = os.path.join(experiment_path_pre, experiment_path_super_res)
        
        if not overwrite and os.path.exists(experiment_path):
            raise ValueError("Overwritting!")

        if mode == "super_res":
            commands.append(
                "python train_super_res.py " + " ".join(f"--{key} {value}" for key, value in setting.items()) + f" --experiment_path {experiment_path}" + f" --mode {mode}")
        else:
            commands.append(
                "python train_dn.py " + " ".join(f"--{key} {value}" for key, value in setting.items()) + f" --experiment_path {experiment_path}" + f" --mode {mode}")

    for command in commands:
        print(command)
    print("Number of commands to execute:", len(commands))

    with open(f"{experiment_path_pre}/_status.tmp", "w") as f:
        now = datetime.datetime.now()
        f.write(now.strftime(f"Experiment: %d/%m/%Y %H:%M:%S\n"))
        f.write(
            "exp,"
            + ",".join(f"{key}" for key, value in settings[0].items())
            + f",timing\n"
        )

        total_time = 0
    for i, (setting, command) in enumerate(zip(settings, commands)):
        print(f"--- Beginning experiment {i} ---")
        start_time = time.process_time()
        subprocess.call(
            command, shell=True
        )  # The experiment will save results on its own
        end_time = time.process_time()
        minutes = (end_time - start_time) / 60
        total_time += minutes

        with open(f"{experiment_path_pre}/_status.tmp", "a+") as f:
            f.write(
                f"{i},"
                + ",".join(f"{value}" for key, value in setting.items())
                + f",{minutes}\n"
            )

    with open(f"{experiment_path_pre}/_status.tmp", "a+") as f:
        f.write(f"Total time taken (minutes): {total_time}")

    print(f"Finished running experiment {experiment_number}")


# ========================================================================
# Super-resolution + MNIST
# ---------------------------------------
# With common randomess models. Change latent_dim, L = (,) accordingly to get tradeoffs at various rates.
settings = []
experiment_number = ("S3")  # This is the folder which will be created to contain the results

mode = "super_res"  # Indicated super-resolution mode
L = 4  # Quantization levels. Controls the rate (R = latent_dim_1*log2(L_1))

# latent_dim_base = [2, 4, 6, 8, 10, 12, 14, 16] # Latent dimensions

latent_dim_base = [16] # Latent dimensions
 
for latent_dim in latent_dim_base:
    settings.append({"L": L, "latent_dim": latent_dim, "epochs": 100, "batch-size": 64, "common": True})

train_with_params(mode, settings, experiment_number, overwrite=False)



# # ========================================================================
# # Denoising + SVHN
# # ---------------------------------------
# # With common randomess models. Change latent_dim, L = (,) accordingly to get tradeoffs at various rates.
# settings = []
# experiment_number = ("D3")  # This is the folder which will be created to contain the results

# mode = "dn"  # Indicated denosing mode
# L = 4  # Quantization levels. Controls the rate (R = latent_dim_1*log2(L_1))

# # latent_dim_base = [2, 4, 6, 8, 10, 12, 14, 16] # Latent dimensions

# latent_dim_base = [6] # Latent dimensions

# for latent_dim in latent_dim_base:
#     settings.append({"L": L, "latent_dim": latent_dim, "epochs": 100, "batch-size": 64, "common": False})

# train_with_params(mode, settings, experiment_number, overwrite=False)