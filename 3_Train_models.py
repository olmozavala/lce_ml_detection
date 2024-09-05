# %%
import os
# External libraries
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, Dataset, random_split

# Local libraries
import multiprocessing as mp
from proj_ai.Training import train_model
from proj_ai.Generators import EddyDataset
from proj_ai.dice_score import dice_loss
from models.ModelsEnum import Models
from models.ModelSelector import select_model
import cmocean.cm as cm
from os.path import join
import time

torch.set_num_threads(128)
# kill %1 
# cmd+z --> pauses
# cmd+c --> kills

#%% ----- DataLoader --------

def parallel_training(gpu, days_before, days_after, input_folder, run="shared", only_ssh=False, start_year=1998, multi_stream=False):
    # 
    # print(f"inside with gpu:{gpu} days_before:{days_before} days_after:{days_after} input_folder:{input_folder} run:{run}")
    # # Wait 5 seconds
    # time.sleep(5)

    # return
    start_time = time.time()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"days_before: {days_before}, days_after: {days_after}, input_folders: {input_folder}")

    bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
    output_resolution = 0.1

    data_folder = "/unity/f1/ozavala/DATA/"
    ssh_folder = join(data_folder, "GOFFISH/AVISO/GoM")
    sst_folder = join(data_folder, "GOFFISH/SST/OSTIA")
    chlora_folder = join(data_folder, "GOFFISH/CHLORA/COPERNICUS")

    eddies_folder = join("/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours_ALL_1993_2022_gaps_filled", input_folder)
    if multi_stream:
        def_output_folder = "/unity/f1/ozavala/OUTPUTS/multi_stream"
    else:
        def_output_folder = "/unity/f1/ozavala/OUTPUTS/single_stream"

    if only_ssh: 
        output_folder = join(def_output_folder,f"EddyDetection_ALL_{start_year}-2022_gaps_filled_submean_only_ssh")
    else:
        output_folder = join(def_output_folder,f"EddyDetection_ALL_{start_year}-2022_gaps_filled_submean_sst_chlora")

    model_name = f"DaysBefore_{days_before}_DaysAfter_{days_after}_{input_folder}"

    print(f"######### Model name: {model_name} #########")

    dataset = EddyDataset(ssh_folder, sst_folder, chlora_folder, eddies_folder, bbox, output_resolution, 
                                      days_before=days_before, days_after=days_after, start_year=start_year, 
                                      only_ssh=only_ssh, multi_stream=multi_stream)

    # Define the sizes for training, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    # Use random_split to split the dataset
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))

    print("Total number of training samples: ", len(train_dataset))
    print("Total number of validation samples: ", len(val_dataset))

    # Create DataLoaders for training and validation
    # TODO the number of workers should be related to the number of processors in the system (90% maybe)
    workers = 2 # For some reason, if I set the number of workers I can't store all the data in memory
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=workers)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=workers)
    print("Done loading data!")

    # Visualize the input data
    # Plot from a batch of training data
    # print("Plotting data...")
    # plot_data_loader= DataLoader(train_dataset, batch_size=1, shuffle=True)
    # dataiter = iter(plot_data_loader)
    # ssh, eddies = next(dataiter)
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # axs[0].imshow(ssh[0,0,:,:], cmap=cm.thermal)
    # axs[1].imshow(eddies[0,0,:,:], cmap='gray')
    # plt.tight_layout()
    # plt.show()
    # plt.title("Example of SSH and eddies contour")
    # print("Done!")

    # Initialize the model, loss, and optimizer
    print("Initializing model...")
    if only_ssh:
        input_channels = 1 + days_before + days_after
    else:
        input_channels = 3 + 3*(days_before+days_after)

    if multi_stream:
        model = select_model(Models.UNET_2D_MultiStream,  num_streams = input_channels, num_levels = 4, cnn_per_level = 2, 
                    input_channels = 1, output_channels = 1, start_filters = 32,
                    kernel_size = 3, batch_norm = True).to(device)
    else:
        model = select_model(Models.UNET_2D, num_levels=4, cnn_per_level=2, input_channels=input_channels,
                         output_channels=1, start_filters=32, kernel_size=3).to(device)


    loss_func = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3000
    print("Training model...")
    train_model(model, optimizer, loss_func, train_loader, val_loader,
                        num_epochs, device, output_folder,
                        model_name=model_name, multi_stream=multi_stream)
    # Print the time elapsed
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Done!")

# %%
# all_input_folders = [f"{x:02d}days_d0_Nx1000_nseeds400" for x in [5, 10, 20, 30]]
# all_input_folders = [f"{x:02d}days_d0_Nx1000_nseeds400" for x in [5, 10, 20, 30]]
all_input_folders = [f"{x:02d}days_d0_Nx1000_nseeds400" for x in [5, 10, 20, 30]]
# all_days_before = [0, 1, 2]
all_days_before = [2]
use_after_days = [True, False]
# only_ssh_opts = [True, False]
only_ssh_opts = [True]
# multi_stream_opts = [False, True]  # Not useful
multi_stream = False
start_year = 1993

# -------------------- Parallel training --------------------
NUM_PROC = 4
i = 0
runs_per_model = 2
# only_ssh_opts = [False]
hd_storage = "shared" # shared local ram

for run in range(runs_per_model):
    for days_before in all_days_before:
        for input_folder in all_input_folders:
            for only_ssh in only_ssh_opts:
                for use_after in use_after_days:
                    gpu = i % NUM_PROC
                    # run WITH plus days the same as the run
                    if use_after:
                        days_after = int(input_folder.split("_")[0].replace("days", ""))
                        p = mp.Process(target=parallel_training, args=(gpu, days_before, days_after, input_folder, hd_storage, only_ssh, start_year, multi_stream))
                    else:
                        p = mp.Process(target=parallel_training, args=(gpu, days_before, 0, input_folder, hd_storage, only_ssh, start_year, multi_stream))
                    p.start()
                    i += 1
        # Wait until all the processes are finished
        p.join()
        print("After join")

# --------------- Sequential training --------------------
# only_ssh = False
# for days_before in all_days_before:
#     for use_after in use_after_days:
#         for input_folder in all_input_folders:
#             # run WITH plus days the same as the run
#                 if use_after:
#                     days_after = int(input_folder.split("_")[0].replace("days", ""))
#                     parallel_training(0, days_before, days_after, input_folder, run=hd_storage, only_ssh = only_ssh, start_year = start_year, multi_stream=multi_stream)
#                 else:
#                     parallel_training(0, days_before, 0, input_folder, run=hd_storage, only_ssh = only_ssh, start_year = start_year, multi_stream=multi_stream)