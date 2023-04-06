# External libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
import numpy as np
# Local libraries
from proj_ai.Training import train_model
from proj_ai.Generators import EddyDataset
from proj_ai.dice_score import dice_coeff, dice_loss
from models.ModelsEnum import Models
from models.ModelSelector import select_model
import cmocean.cm as cm
from os.path import join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("Using device: ", device)

#%% ----- DataLoader --------
# ssh_folder = "/data/LOCAL_GOFFISH/AVISO/"
# eddies_folder = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/GOFFISH_UGOS3/ProgramsRepo/data/Geodesic/processed/"
data_folder = "/unity/f1/ozavala/DATA/GOFFISH/"
ssh_folder = join(data_folder, "/GOFFISH/AVISO")
eddies_folder = join(data_folder,"/GOFFISH/EddyDetection/PreprocContours/")
eddies_folder = join(data_folder, "/EddyDetection/PreprocContours/")
output_folder = join(data_folder, "/unity/f1/ozavala/CODE/EddyDetection")

bbox = [18.125, 32.125, 260.125 - 360, 285.125 - 360]  # These bbox needs to be the same used in preprocessing
output_resolution = 0.1
train_perc = 0.8
val_perc = 0.2

train_dataset = EddyDataset(ssh_folder, eddies_folder, bbox, output_resolution)

train_size = int(train_perc * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
print("Total number of training samples: ", len(train_dataset))
print("Total number of validation samples: ", len(val_dataset))

# Create DataLoaders for training and validation
workers = 20
train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=10, num_workers=workers)
print("Done loading data!")

#%% Visualize the data
# Plot from a batch of training data
dataiter = iter(train_loader)
ssh, eddies = next(dataiter)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(ssh[0,0,:,:], cmap=cm.thermal)
axs[1].imshow(eddies[0,0,:,:], cmap='gray')
plt.tight_layout()
plt.show()
plt.title("Example of SSH and eddies contour")
print("Done!")

#%% Initialize the model, loss, and optimizer
model = select_model(Models.UNET_2D, num_levels=4, cnn_per_level=2, input_channels=1,
                     output_channels=1, start_filters=32, kernel_size=3).to(device)

# TODO how to parallelize the training in multiple GPUs
# n_gpus = torch.cuda.device_count()
# torch.distributed.init_process_group( backend='nccl', world_size=N, init_method='...')
# model = DistributedDataParallel(model, device_ids=[i], output_device=i)

# criterion = nn.MSELoss()
loss_func = dice_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
model = train_model(model, optimizer, loss_func, train_loader, val_loader, num_epochs, device, output_folder)

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
#
# #%% Showing wrong labels
# ouput_value = output.argmax(dim=1).cpu()
# dif = np.where(ouput_value != target.cpu())[0]
# cur_idx = 0
# #%%
# fig, ax = plt.subplots(plot_n, 1, figsize=(5, 5*plot_n))
# for i in range(plot_n):
#     ax[i].imshow(data[dif[cur_idx]].to('cpu').numpy().squeeze(), cmap='gray')
#     ax[i].set_title(f'True: {target[dif[cur_idx]]}, Prediction: {output[dif[cur_idx]].argmax(dim=0)} {[f"{x:0.2f}" for x in output[dif[cur_idx]]]}', wrap=True)
#     cur_idx += 1
# plt.show()
# print("Done!")