import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from os.path import join
from eoas_pyutils.io_utils.io_common import create_folder
import torch.optim.lr_scheduler as lr_scheduler

def train_model(model, optimizer, loss_func, train_loader, val_loader, num_epochs, device, output_folder='training', model_name=''):
    '''
    Main function in charge of training a model
    :param model:
    :param optimizer:
    :param loss_func:
    :param train_loader:
    :param val_loader:
    :param num_epochs:
    :param device:
    :return:
    '''
    cur_time = datetime.now()
    model_name = f'{model_name}_{cur_time.strftime("%Y-%m-%d_%H_%M")}'
    logs_folder = join(output_folder, 'logs', model_name)
    models_folder = join(output_folder, 'models', model_name)
    create_folder(output_folder)
    create_folder(models_folder)
    create_folder(logs_folder)

    writer = SummaryWriter(logs_folder)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    min_val_loss = 1e10
    for epoch in range(num_epochs):
        model.train()
        # Loop over each batch from the training set (to update the model)
        for file, (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        cur_val_loss = 0
        correct = 0
        # Loop over each batch from the test set (to evaluate the model)
        with torch.no_grad():
            for file, (data, target) in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                cur_val_loss += loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        cur_val_loss /= len(val_loader.dataset)
        # scheduler.step(cur_val_loss)
        if cur_val_loss < min_val_loss:
            min_val_loss = cur_val_loss
            torch.save(model.state_dict(), join(models_folder, f'epoch_{epoch:02d}_loss_{min_val_loss:0.4f}.pt'))

        writer.add_scalar('Loss/train', loss/len(train_loader.dataset), epoch)
        writer.add_scalar('Loss/val', cur_val_loss, epoch)
        # writer.add_scalars('train/val', {'training':loss/len(train_loader.dataset), 'validation':cur_val_loss}, global_step=epoch)
        writer.add_scalars('train/val', {'training':loss, 'validation':cur_val_loss}, global_step=epoch)

        _, (images, labels) = next(iter(val_loader))
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        # Here we should add some input output images to the tensorboard
        imgs_to_show = 4
        grid_images = torchvision.utils.make_grid(images[:imgs_to_show, :, :, :], nrow=imgs_to_show)  # 4 images per row
        grid_labels = torchvision.utils.make_grid(labels[:imgs_to_show, :, :, :], nrow=imgs_to_show)  # 4 images per row
        grid_outputs = torchvision.utils.make_grid(output[:imgs_to_show, :, :, :], nrow=imgs_to_show)  # 4 images per row
        writer.add_image('input (validation)', grid_images, epoch)
        writer.add_image('target (validation)', grid_labels, epoch)
        writer.add_image('output (validation)', grid_outputs, epoch)
        if epoch == 0:
            writer.add_graph(model, images)

        print(f'Epoch: {epoch+1}, Val loss: {cur_val_loss:.4f}')

    print("Done!")
    writer.close()
    return model