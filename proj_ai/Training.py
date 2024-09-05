import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from os.path import join
from eoas_pyutils.io_utils.io_common import create_folder
import torch.optim.lr_scheduler as lr_scheduler


def train_model(model, optimizer, loss_func, train_loader, val_loader, num_epochs, device, 
                output_folder='training', model_name='', multi_stream=False):

    cur_time = datetime.now()
    model_name = f'{model_name}_{cur_time.strftime("%Y-%m-%d_%H_%M")}'
    logs_folder = join(output_folder, 'logs', model_name)
    models_folder = join(output_folder, 'models', model_name)
    create_folder(output_folder)
    create_folder(models_folder)
    create_folder(logs_folder)

    writer = SummaryWriter(logs_folder)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, verbose=True)

    min_val_loss = 1e10
    best_last_epoch = 0
    patience = 40 # Number of epochs to wait before stopping the training
    for epoch in range(num_epochs):
        # Time each epoch
        start_time_epoch = datetime.now()
        model.train()
        # Loop over each batch from the training set (to update the model)
        for file, (data, target) in train_loader:
            # Time each batch
            # print(f"Number of thredds: {torch.get_num_threads()}")
            start_time_batch = datetime.now()

            if multi_stream:
                # -------------- MultiStream 
                # data should be a list of tensors (one per stream), each stream shape is (batch_size, 1, H, W)
                data, target = tuple(tensor.to(device) for tensor in data),  target.to(device)  # MultiStream
                # data = tuple(torch.rand(128, 1, 64, 64).to(device) for _ in range(3))
                optimizer.zero_grad()
                output = model(*data)
            else: 
                # -------------- SingleStream
                data, target = data.to(device), target.to(device)  
                optimizer.zero_grad()
                output = model(data)

            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            # print(f'Batch time: {datetime.now() - start_time_batch}')

        # Evaluate the model
        model.eval()
        cur_val_loss = 0
        correct = 0
        # Loop over each batch from the test set (to evaluate the model)
        with torch.no_grad():
            for file, (data, target) in val_loader:
                if multi_stream:
                    # -------------- MultiStream 
                    # data should be a list of tensors (one per stream), each stream shape is (batch_size, 1, H, W)
                    data, target = tuple(tensor.to(device) for tensor in data),  target.to(device)  # MultiStream
                    # data = tuple(torch.rand(128, 1, 64, 64).to(device) for _ in range(3))
                    output = model(*data)
                else: 
                    # -------------- SingleStream
                    data, target = data.to(device), target.to(device)  
                    output = model(data)

                cur_val_loss += loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        cur_val_loss /= len(val_loader.dataset)
        scheduler.step(cur_val_loss)
        # Save the model if it is the best one and save example images
        if cur_val_loss < min_val_loss:
            best_last_epoch = epoch
            min_val_loss = cur_val_loss
            torch.save(model.state_dict(), join(models_folder, f'epoch_{epoch:02d}_loss_{min_val_loss:0.4f}.pt'))

            if multi_stream:
                _, (images, labels) = next(iter(val_loader))
                images = tuple(tensor.to(device) for tensor in images)
                labels = labels.to(device)
                output = model(*images)
            else:
                # Just save images when the model is improved
                _, (images, labels) = next(iter(val_loader))
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                # Here we should add some input output images to the tensorboard
                imgs_to_show = 4
                grid_labels = torchvision.utils.make_grid( torch.flip(labels[:imgs_to_show, :, :, :], dims=[2]), nrow=imgs_to_show)  # 4 images per row
                grid_outputs = torchvision.utils.make_grid(torch.flip(output[:imgs_to_show, :, :, :], dims=[2]), nrow=imgs_to_show)  # 4 images per row
                writer.add_image('target (validation)', grid_labels, epoch)
                writer.add_image('output (validation)', grid_outputs, epoch)

            writer.add_scalar('Loss/train', loss/len(train_loader.dataset), epoch)
            writer.add_scalar('Loss/val', cur_val_loss, epoch)

            # if epoch == 0:
            #     total_inputs = images.shape[1]
            #     grid_images = torchvision.utils.make_grid( torch.flip(images[:, :, :, :], dims=[1]), nrow=imgs_to_show)  # 4 images per row (only show one date)
            #     writer.add_image('inputs', grid_images, epoch)

        if epoch == 0:
            writer.add_graph(model, images)

        print(f'------------ Epoch: {epoch+1}, Val loss: {cur_val_loss:.4f} Time: {datetime.now() - start_time_epoch}')

        if epoch - best_last_epoch > patience:
            print(f'Early stopping after {epoch} epochs. CurEpoch {epoch}, BestEpoch {best_last_epoch} (should be bigger than {patience})')
            break

    print("Done!")
    writer.close()
    # Write a placeholder file to indicate that the training is done with tthe current time and best model found
    with open(join(models_folder, 'training_done.txt'), 'w') as f:
        f.write(f'Training done at {datetime.now()} with best model at epoch {best_last_epoch} with loss {min_val_loss} (ended at epoch {epoch})')

    return model