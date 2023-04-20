import os
# External libraries
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# Local libraries
import torch.multiprocessing as mp
import multiprocessing as mp
from proj_ai.Training import train_model
from proj_ai.Generators import EddyDataset
from proj_ai.dice_score import dice_loss
from models.ModelsEnum import Models
from models.ModelSelector import select_model
import cmocean.cm as cm
from os.path import join
import time

#%% ----- DataLoader --------

def parallel_training(gpu, days_before, days_after, input_folders):
    start_time = time.time()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"days_before: {days_before}, days_after: {days_after}, input_folders: {input_folders}")

    data_folder = "/unity/f1/ozavala/DATA/GOFFISH"
    bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
    output_resolution = 0.1
    val_perc = 90 / 365  # For year 2020 how many files to use for validation 2 months =

    val_folder = '2020_coh_14days_contour_d7'
    ssh_folder = join(data_folder, "AVISO")

    eddies_folder = [join(data_folder, f"EddyDetection/PreprocContours_{folder}") for folder in input_folders]
    eddies_folder_val = [join(data_folder, f"EddyDetection/PreprocContours_{val_folder}")]
    output_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection"

    if type(input_folders) is list:
        model_name = f"DaysBefore_{days_before}_DaysAfter_{days_after}_{'_'.join(input_folders)}"
    else:
        model_name = f"DaysBefore_{days_before}_DaysAfter_{days_after}_{input_folders}"

    print(f"######### Model name: {model_name} #########")

    train_dataset = EddyDataset(ssh_folder, eddies_folder, bbox, output_resolution, days_before=days_before, days_after=days_after)
    val_dataset = EddyDataset(ssh_folder, eddies_folder_val, bbox, output_resolution, perc_files=val_perc,
                              days_before=days_before, days_after=days_after)

    print("Total number of training samples: ", len(train_dataset))
    print("Total number of validation samples: ", len(val_dataset))

    # Create DataLoaders for training and validation
    # workers = 30  # For some reason, if I set the number of workers I can't store all the data in memory
    # train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=workers)
    # val_loader = DataLoader(val_dataset, batch_size=10, num_workers=workers)
    train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)
    print("Done loading data!")

    #%% Visualize the data
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

    #%% Initialize the model, loss, and optimizer
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

if __name__ == '__main__':
    # ['2010_coh_14days_contour_d7', '2015_coh_14days_contour_d7', '2018_coh_14days_contour_d7', '2020_coh_14days_contour_d7', '2020_coh_14days_contour_d0']
    all_input_folders = [['2010_coh_14days_contour_d7', '2015_coh_14days_contour_d7', '2018_coh_14days_contour_d7'],
                         ['2010_coh_14days_contour_d7', '2015_coh_14days_contour_d7'],
                         ['2010_coh_14days_contour_d7']]
    # all_days_before = [0, 6]
    # all_days_after = [0, 6]
    all_days_before = [0]
    all_days_after = [0]

    # Parallelizing by input folders

    NUM_PROC = 1
    mp.set_start_method('spawn')

    for gpu in range(NUM_PROC):
        for days_before in all_days_before:
            for days_after in all_days_after:
                # Parallel trianing (only by folder for now)
                p = mp.Process(target=parallel_training, args=(gpu, days_before, days_after, all_input_folders[gpu]))
                p.start()

            # Sequential training
            # for input_folders in all_input_folders:
            #     parallel_training(days_before, days_after, input_folders)