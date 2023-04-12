# External libraries
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# Local libraries
from proj_ai.Training import train_model
from proj_ai.Generators import EddyDataset
from proj_ai.dice_score import dice_loss
from models.ModelsEnum import Models
from models.ModelSelector import select_model
import cmocean.cm as cm
import os
from os.path import join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("Using device: ", device)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#%% ----- DataLoader --------
data_folder = "/unity/f1/ozavala/DATA/GOFFISH"
folders = ['2010_coh_14days_contour_d7', '2020_coh_14days_contour_d7', '2020_coh_14days_contour_d0']
c_folder = folders[0]
val_folder = folders[1]

ssh_folder = join(data_folder, "AVISO")
eddies_folder = join(data_folder,f"EddyDetection/PreprocContours_{c_folder}")
eddies_folder_val = join(data_folder,f"EddyDetection/PreprocContours_{val_folder}")
output_folder = "/unity/f1/ozavala/OUTPUTS/EddyDetection"

bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
output_resolution = 0.1
val_perc = 60/365  # For year 2020 how many files to use for validation 2 months =

train_dataset = EddyDataset(ssh_folder, eddies_folder, bbox, output_resolution)
val_dataset = EddyDataset(ssh_folder, eddies_folder_val, bbox, output_resolution, perc_files=val_perc)

print("Total number of training samples: ", len(train_dataset))
print("Total number of validation samples: ", len(val_dataset))

# Create DataLoaders for training and validation
# workers = 30  # For some reason, if I set the number of workers I can't store all the data in memory
# train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=workers)
# val_loader = DataLoader(val_dataset, batch_size=10, num_workers=workers)
train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10)
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
model = select_model(Models.UNET_2D, num_levels=4, cnn_per_level=2, input_channels=1,
                     output_channels=1, start_filters=32, kernel_size=3).to(device)

loss_func = dice_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3000
model_name = c_folder
print("Training model...")
model = train_model(model, optimizer, loss_func, train_loader, val_loader,
                    num_epochs, device, output_folder,
                    model_name=model_name)
print("Done!")

# #%% Show the results
# # Get a batch of test data
# plot_n = 2
# dataiter = iter(val_loader)
# data, target = next(dataiter)
# data, target = data.to(device), target.to(device)
# output = model(data)
# fig, ax = plt.subplots(plot_n, 1, figsize=(5, 5*plot_n))
# for i in range(plot_n):
#     ax[i].imshow(data[i].to('cpu').numpy().squeeze(), cmap='gray')
#     ax[i].set_title(f'True: {target[i]}, Prediction: {output[i].argmax(dim=0)} {[f"{x:0.2f}" for x in output[i]]}', wrap=True)
# plt.show()
# print("Done!")