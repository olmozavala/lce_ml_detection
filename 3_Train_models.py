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

def parallel_training(gpu, days_before, days_after, input_folder, run="shared"):
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

    aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM"
    eddies_folder = join("/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours_ALL_1993_2022_gaps_filled", input_folder)
    output_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection_ALL_1993_2022_gaps_filled_submean"
    # if run == "shared":
    #     aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM"
    #     eddies_folder = join("/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours_ALL_1993_2022_gaps_filled", input_folder)
    #     # output_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection_ALL_1993_2022"
    #     output_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection_ALL_1993_2022_gaps_filled"
    # elif run == "local":
    #     aviso_folder = "/conda/ozavala/data/GoM"
    #     eddies_folder = join("/conda/ozavala/data/PreprocContours_ALL_1993_2022", input_folder)
    #     # output_folder = "/conda/ozavala/data/OUTPUTS/EddyDetection_ALL_1993_2022"
    #     output_folder = "/conda/ozavala/data/OUTPUTS/EddyDetection_ALL_1993_2022_gaps_filled"
    # elif run == "ram":
    #     aviso_folder = "/dev/shm/ozavala/data/GoM"
    #     eddies_folder = join("/dev/shm/ozavala/data/PreprocContours_ALL_1993_2022", input_folder)
    #     # output_folder = "/dev/shm/ozavala/data/OUTPUTS/EddyDetection_ALL_1993_2022"
    #     output_folder = "/dev/shm/ozavala/data/OUTPUTS/EddyDetection_ALL_1993_2022_gaps_filled"

    model_name = f"DaysBefore_{days_before}_DaysAfter_{days_after}_{input_folder}"

    print(f"######### Model name: {model_name} #########")

    dataset = EddyDataset(aviso_folder, eddies_folder, bbox, output_resolution, days_before=days_before, days_after=days_after)

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

    # Visualize the data
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
    model = select_model(Models.UNET_2D, num_levels=4, cnn_per_level=2, input_channels=1+days_before+days_after,
                         output_channels=1, start_filters=32, kernel_size=3).to(device)

    

    loss_func = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3000
    print("Training model...")
    model = train_model(model, optimizer, loss_func, train_loader, val_loader,
                        num_epochs, device, output_folder,
                        model_name=model_name)
    # Print the time elapsed
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Done!")


# %%
all_input_folders = [f"{x:02d}days_d0_Nx1000_nseeds400" for x in [5, 10, 20, 30]]
all_days_before = [0, 1, 2]
use_after_days = [False, True]

# -------------------- Parallel training --------------------
NUM_PROC = 4
i = 0
runs_per_model = 2
for run in range(runs_per_model):
    for use_after in use_after_days:
        for days_before in all_days_before:
            for input_folder in all_input_folders:
                gpu = i % NUM_PROC
                # run WITH plus days the same as the run
                if use_after:
                    days_after = int(input_folder.split("_")[0].replace("days", ""))
                    p = mp.Process(target=parallel_training, args=(gpu, days_before, days_after, input_folder, "shared"))
                else:
                    p = mp.Process(target=parallel_training, args=(gpu, days_before, 0, input_folder, "shared"))
                p.start()
                i += 1
        # Wait until all the processes are finished
        p.join()
        print("After join")

# --------------- Sequential training --------------------
# source = "local" # shared local ram
# for days_before in all_days_before:
#     for use_after in use_after_days:
#         for input_folder in all_input_folders:
#             # run WITH plus days the same as the run
#                 if use_after:
#                     days_after = int(input_folder.split("_")[0].replace("days", ""))
#                     parallel_training(0, days_before, days_after, input_folder, run=source)
#                 else:
#                     parallel_training(0, days_before, 0, input_folder, run=source)