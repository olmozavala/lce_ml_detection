import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# AI common
import sys
sys.path.append("ai_common")
from ai_common.constants.AI_params import TrainingParams, EvaluationParams

# EOAS Utils
sys.path.append("eoas_pyutils")
from eoas_pyutils.io_utils.io_common import create_folder

# This Project
from config.MainConfig import get_config


# def makeScatter(c_summary, group_field, xlabel, output_file):
#     """
#     Makes a scatter plot from a dataframe grouped by the specified "group_field"
#     :param c_summary:
#     :param group_field:
#     :param xlabel:
#     :param output_file:
#     :return:
#     """
#     LOSS  = "Loss value"
#     names = []
#     data = []
#     fig, ax = plt.subplots(figsize=(10,6))
#     for i, grp in enumerate(c_summary.groupby(group_field)):
#         names.append(grp[0])
#         c_data = grp[1][LOSS].values
#         data.append(c_data)
#         plt.scatter(np.ones(len(c_data))*i, c_data, label=grp[0])
#
#     plt.legend(loc="best")
#     # bp = plt.boxplot(data, labels=names, patch_artist=True, meanline=True, showmeans=True)
#     ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel("Validation Loss (MSE)")
#     ax.set_title(F"Validation Loss by {group_field} Type ")
#     plt.savefig(join(output_folder,output_file))
#     plt.show()

if __name__ == '__main__':

    NET = "Network Type"
    LOSS  = "Loss value"

    # Read folders for all the experiments
    config = get_config()
    trained_models_folder = config[TrainingParams.output_folder]
    output_folder = config[EvaluationParams.output_folder]

    all_folders = os.listdir(trained_models_folder)
    all_folders.sort()

    experiments = []

    # Iterate over all the experiments
    for experiment in all_folders:
        models_folder = join(trained_models_folder, experiment , "models")
        if os.path.exists(models_folder):
            print(F"Working with experiment: {experiment}")
            all_models = os.listdir(models_folder)
            min_loss = 100000.0
            best_model = {}
            # Iterate over the saved models for each experiment and obtain the best of them
            for model in all_models:
                model_split = model.split("-")
                loss = float((model_split[-1]).replace(".hdf5",""))
                if loss < min_loss:
                    min_loss = loss
                    id = model_split[0]
                    name = model_split[1]
                    network_type = model_split[2]
                    loss = np.around(min_loss,5)
                    path = join(trained_models_folder, experiment, "models", model)
                    best_model = [id, experiment, network_type, loss, path]
            experiments.append(best_model)

    # Build a dictionary from data
    df = {
        "ID": [x[0] for x in experiments],
        "Name": [x[1] for x in experiments],
        "Network Type": [x[2] for x in experiments],
        "Loss value": [x[3] for x in experiments],
        "Path": [x[4] for x in experiments],
    }
    summary = pd.DataFrame.from_dict(df)
    print(F"Models summary: {summary}")

    create_folder(output_folder)
    summary.to_csv(join(output_folder,"summary.csv"))

    # def_bbox = "384x520"
    # def_in = "ssh"
    # def_out = "SRFHGT"
    # def_perc_ocean = "0.0"
    #
    # # ========= Compare Network type ======
    # c_summary = summary[np.logical_and((summary[IN] == def_in).values, (summary[OUT] == def_out).values)]
    # c_summary = c_summary[c_summary["BBOX"] == def_bbox]
    # c_summary = c_summary[c_summary[PERCOCEAN] == def_perc_ocean]  # Only PercOcean 0.0
    # makeScatter(c_summary, NET, "Network type", "By_Network_Type_Scatter.png")