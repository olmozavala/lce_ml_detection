# External
import os
from pandas import DataFrame
import pandas as pd
import time
from os.path import join
import numpy as np
import xarray as xr
from tensorflow.keras.utils import plot_model

# Common
import sys
# AI common
sys.path.append("ai_common")
from ai_common.constants.AI_params import TrainingParams, EvaluationParams, ModelParams
from ai_common.models.modelSelector import select_2d_model
import ai_common.training.trainingutils as utilsNN
# EOAS Utils
sys.path.append("eoas_pyutils")
from eoas_pyutils.viz_utils.eoa_viz import EOAImageVisualizer
from eoas_pyutils.io_utils.io_common import create_folder, all_files_in_folder

# From project
from config.MainConfig import get_config
from proj_io.io_reader import generateXY

from sklearn.metrics import *


config = get_config()
"""
:param config:
:return:
"""
# *********** Reads the parameters ***********
output_folder = config[EvaluationParams.output_folder]
summary_file = join(output_folder, "summary.csv")
df = pd.read_csv(summary_file)

save_predictions = False
save_imgs = True

#  ========= Obtain the total number of examples (this most be the same as in the training phase) ============
input_folder = config[TrainingParams.input_folder]
lr_names, lr_paths = all_files_in_folder(join(input_folder, 'GOMb0.04'), file_ext="2d.nc")
lr_names.sort()
lr_paths.sort()
hr_names, hr_paths = all_files_in_folder(join(input_folder, 'GOMb0.01'), file_ext="2d.nc")
hr_names.sort()
hr_paths.sort()
tot_examples = len(lr_names)

# ============== Configurations needed ================
in_dims = config[ModelParams.INPUT_SIZE]
inc_factor = config[ModelParams.INC_RES_FACTOR]

# Iterates over all the models in the file
for model_id in range(len(df)):
    model = df.iloc[model_id]
    model_name = model["Name"]
    model_weights_file = model["Path"]
    print(F"Model model_name: {model_name}")

    c_output_folder =join(output_folder,model_name)
    create_folder(c_output_folder)

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    model = select_2d_model(config)
    plot_model(model, to_file=join(c_output_folder,F'{model_name}.png'), show_shapes=True)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    # **************** Split definition *****************
    SEED = 0
    np.random.seed(SEED)  # Use this to always train with the same split for all the models
    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(tot_examples,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc)

    # Working only with the test indexes
    for c_id in range(len(test_ids)):
        file_id = test_ids[c_id]
        # TODO here you could test in batches also
        X, Y = generateXY(hr_path=hr_paths[file_id],
                          lr_path=lr_paths[file_id],
                          inc_factor=inc_factor,
                          in_dims=in_dims)

        X = np.expand_dims(X, axis=0)
        nn_raw_output = model.predict(X, verbose=1)
        #
        lr_ds = xr.open_dataset(lr_paths[file_id])
        lr_ds_crop = lr_ds.isel(Latitude=slice(0, in_dims[0]), Longitude=slice(0, in_dims[1]), MT=0)  # Cropping by value
        lats = lr_ds_crop.Latitude
        lons = lr_ds_crop.Longitude
        vizobj = EOAImageVisualizer(disp_images=False, output_folder=c_output_folder, lats=[lats],lons=[lons], show_var_names=True)
        vizobj.plot_2d_data_np(np.swapaxes(X[0], 0, 2), ['U', 'V'], F'X_{lr_names[file_id]}_lr', file_name_prefix=F'X_{lr_names[file_id]}_lr', flip_data=True, rot_90=True)
        vizobj.plot_2d_data_np(np.swapaxes(nn_raw_output[0], 0, 2), ['U', 'V'], F'Y_{hr_names[file_id]}_hr', file_name_prefix=F'Y_{hr_names[file_id]}_hr', flip_data=True, rot_90=True)

    # *********** Makes a dataframe to contain the DSC information **********
    # metrics_params = config[ClassificationParams.metrics]
    # metrics_dict = {met.model_name: met.value for met in metrics_params}

    # ------------------- Making prediction -----------
    # print('\t Making prediction....')
    # X = 1
    # nn_raw_output = model.predict(X, verbose=1)
    #
    # mse_mft = mean_squared_error(mft,obs)
    # mse_nn = mean_squared_error(output_nn_all_original,obs)
    # print(F'MSE MFT: {mse_mft}  NN: {mse_nn}')
    #
    # if save_predictions:
    #     print('\t Saving Prediction...')
    # if save_imgs:
    #     print('\t Saving Images...')
