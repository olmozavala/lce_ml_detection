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
models_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection/models"
output_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection/Summary"

all_folders = os.listdir(models_folder)
all_folders.sort()

experiments = []

# ----------- Build a summary of the models ------------
# Iterate over all the experiments
for cur_model in all_folders:
    cur_folder = join(models_folder, cur_model)
    if os.path.exists(cur_folder):
        print(F"Working with cur_model: {cur_model}")
        all_models = os.listdir(cur_folder)
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
                best_model = [name, loss, path]
        experiments.append(best_model)

# Build a dictionary from data
df = {
    "Name": [x[0] for x in experiments],
    "Loss value": [x[1] for x in experiments],
    "Path": [x[2] for x in experiments],
}
summary = pd.DataFrame.from_dict(df)
print(F"Models summary: {summary}")

create_folder(output_folder)
summary = summary.sort_values(by="Loss value")
summary.to_csv(join(output_folder,"summary.csv"))

#%% ------- Plot the summary of the models ------------
output_file = join(output_folder,"summary.jpg")
# plot_summary(summary)
LOSS  = "Loss value"
names = summary.Name.values
names = ['_'.join(x.split('_')[:4]) for x in summary.Name.values]
losses = summary[LOSS].values
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(range(len(names)), losses)
ax.set_title(F"Validation Loss by Model")
for i, label in enumerate(names):
    ax.annotate(label, (i, losses[i]), textcoords="offset points",
                xytext=(5, -15), ha='left', va='center', rotation=0, fontsize=12)

ax.set_xlim(-0.5, len(names)+0.5)
ax.grid(True)
ax.set_ylabel("Validation Loss (MSE)")
plt.savefig(join(output_file))
plt.show()
