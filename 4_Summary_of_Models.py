# %%
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# AI common
import sys
sys.path.append("ai_common")
# EOAS Utils
sys.path.append("eoas_pyutils")
from eoas_pyutils.io_utils.io_common import create_folder

#%% ------ Define paths --------
models_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection_ALL_1993_2022_gaps_filled_submean/models"
output_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection_ALL_1993_2022_gaps_filled_submean/Summary"

all_folders = os.listdir(models_folder)
all_folders.sort()

experiments = []

# ----------- Build a summary of the models ------------
# Iterate over all the experiments
for cur_model in all_folders:
    cur_folder = join(models_folder, cur_model)
    if os.path.exists(cur_folder):
        print(F"Working with cur_model: {cur_model}")
        all_models = [x for x in os.listdir(cur_folder) if x.endswith(".pt")]
        min_loss = 100000.0
        best_model = {}
        # Iterate over the saved models for each cur_model and obtain the best of them
        for model in all_models:
            model_split = model.split("_")
            loss = float((model_split[-1]).replace(".pt",""))
            if loss < min_loss:
                min_loss = loss
                name = cur_model
                loss = np.around(min_loss,5)
                path = join(cur_folder, model)
                days_before = int(name.split('_')[1])
                days_after = int(name.split('_')[3])
                lcv_length = int((name.split('_')[4]).replace("days",""))
                best_model = [name, loss, path, days_before, days_after, lcv_length]
        print(best_model)
        experiments.append(best_model)

# Build a dictionary from data
LOSS = "Loss value"
LCV_LEN = "LCV length"

# Remove empty experiments
experiments = [x for x in experiments if len(x) > 0]

df = {
    "Name": [x[0] for x in experiments],
    LOSS: [x[1] for x in experiments],
    "Path": [x[2] for x in experiments],
    "Days before": [x[3] for x in experiments],
    "Days after": [x[4] for x in experiments],
    LCV_LEN: [x[5] for x in experiments],
}
summary = pd.DataFrame.from_dict(df)
print(F"Models summary: {summary}")

create_folder(output_folder)
summary = summary.sort_values(by="Loss value")
summary.to_csv(join(output_folder,"summary.csv"))

#%% ------- Plot the summary of the models ------------
output_file = join(output_folder,"summary.jpg")
fig, ax = plt.subplots(figsize=(10,6))
grouped = summary.groupby(LCV_LEN)
for group_key, group_data in grouped:
    print(group_key, group_data)
    names = group_data.Name.values
    names = [f'-{x.split("_")[1]} +{x.split("_")[3]}' for x in names]
    losses = group_data[LOSS].values
    ax.scatter(range(len(losses)), losses, label=group_key)

    for i, label in enumerate(names):
        ax.annotate(label, (i, losses[i]), textcoords="offset points",
                    xytext=(5, -15), ha='left', va='center', rotation=0, fontsize=12)

ax.legend()
ax.set_title(F"Best Validation Loss by Model Run", fontsize=16)
# ax.set_xlim(-0.5, len(names)-0.2)
# ax.set_ylim(0.052, 0.078)
# ax.grid(True)
ax.set_ylabel("Validation Loss (MSE)", fontsize=14)
ax.set_xlabel("Multiple trainings sorted by validation error", fontsize=14)
plt.savefig(join(output_file))
plt.show()
# %%
