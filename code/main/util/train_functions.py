# Import Libraries
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

from attention_model import ResNet_extractor
from loaders import default_loader
from preprocess import read_ldmb

from timeit import default_timer as timer

# set up logging
import logging
logger = logging.getLogger('tcga_logger.' + __name__)


# function to train model and display accuracy, loss for train and val set   
# code adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=1,
          scheduler=None):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
        logging.info(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')
        logging.info(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):
        
        # in case of multiple dataloaders (for multiple undersampled versions of the train set, see thesis section 6.3.3, "undersampling the dataset")
        if isinstance(train_loader, list):
            train_loader_ep = train_loader[epoch % 30]
        else:
            train_loader_ep = train_loader

        if isinstance(valid_loader, list):
            valid_loader_ep = valid_loader[epoch % 30]
        else:
            valid_loader_ep = valid_loader

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0
        valid_auc = 0
        amount_not_in_auc = 0  # in the final batch it can happen that there is only one label -> can't calculate auc on these, in this case this variable = batch size, else 0

        # Set to training
        model.train()
        start = timer()
        
        # Decay Learning Rate
        if scheduler is not None:
            scheduler.step()
            # Print Learning Rate
            # print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
            # logging.info("Epoch: {0}".format(epoch))
            # logging.info("Learning rate: {0}".format(scheduler.get_last_lr()))

        # Training loop
        for ii, (data, target) in enumerate(train_loader_ep):
            # Tensors to gpu
            data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            
            # BCE expects only one output
            target = target.unsqueeze(1)
            target = target.float()  
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)
            
            # for BCEWithLogitsLoss we need to add sigmoid in inference
            pred = torch.round(torch.sigmoid(output)) 
                
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            if (ii % 5 == 0):
                # Track training progress
                print(
                    f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader_ep):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')
        #print('\n')
            
        # After training loops ends, start validation
        model.epochs += 1

        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            # Validation loop
            for index, (data, target) in enumerate(valid_loader_ep):
                # Tensors to gpu
                data, target = data.cuda(), target.cuda()

                target = target.unsqueeze(1)
                target = target.float()

                output = model(data)
                loss = criterion(output, target)

                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)

                # Calculate validation accuracy
                # for BCEWithLogitsLoss we need to add sigmoid in inference
                prob = torch.sigmoid(output)
                pred = torch.round(torch.sigmoid(output))

                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

                # Multiply average accuracy times the number of examples
                valid_acc += accuracy.item() * data.size(0)
                
                if len(np.unique(target.cpu().numpy().flatten())) > 1:
                    auc = roc_auc_score(target.cpu().numpy().flatten(), prob.cpu().numpy().flatten())
                    valid_auc += auc * data.size(0)
                else:  # in the final batch it can happen that there is only one label (e.g. 4 times 0 -> can't calculate auc)
                    amount_not_in_auc = data.size(0)

                if (index % 5 == 0):
                    # Track training progress
                    print(
                        f'Epoch: {epoch}\t{100 * (index + 1) / len(valid_loader_ep):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                        end='\r')

            # Calculate average losses
            train_loss = train_loss / len(train_loader_ep.dataset)
            valid_loss = valid_loss / len(valid_loader_ep.dataset)

            # Calculate average accuracy
            train_acc = train_acc / len(train_loader_ep.dataset)
            valid_acc = valid_acc / len(valid_loader_ep.dataset)
            
            # Calculate auc
            valid_auc = valid_auc / (len(valid_loader_ep.dataset)-amount_not_in_auc)

            history.append([train_loss, valid_loss, train_acc, valid_acc, valid_auc])

            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                )
                print(
                    f'\t\tValidation AUC: {valid_auc:.4f}'
                )
                logging.info(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                )
                logging.info(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                )
                logging.info(
                    f'\t\tValidation AUC: {valid_auc:.4f}'
                )

            # Save the model if validation loss decreases
            if valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)

                state = {
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()    
                }
                if scheduler != None:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, save_file_name.replace('-.pt','-state.pt')) # save optimizer and nr epochs

                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch

            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                    )
                    total_time = timer() - overall_start
                    print(
                        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                    )
                    logging.info(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                    )
                    logging.info(
                        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                    )

                    # Load the best state dict
                    model.load_state_dict(torch.load(save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer

                    # Format history
                    history = pd.DataFrame(
                        history,
                        columns=[
                            'train_loss', 'valid_loss', 'train_acc',
                            'valid_acc', 'valid_auc'
                        ])
                    return model, history
    
    # Load the best state dict
    model.load_state_dict(torch.load(save_file_name))
    
    # Attach the optimizer
    model.optimizer = optimizer
    
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%'
    )
    logging.info(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%'
    )
    
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'valid_auc'])
    return model, history


class ImageDataset(Dataset):
    """
    custom dataset class to load images
    """
    def __init__(self, tile_paths_list, transform=None, db_path=None, mapping_dict=None):
        # self.df = df
        self.transform = transform
        self.tile_paths_list = tile_paths_list
        self.db_path = db_path
        self.mapping_dict = mapping_dict

    def __len__(self):
        return len(self.tile_paths_list)

    def __getitem__(self, index):
        img_path = self.tile_paths_list[index]

        if self.db_path == None:
            image = default_loader(img_path)
        else:
            img_name = img_path.split('/')[-1]
            slide_name = img_name.split('_')[0]
            path = self.db_path+'/'+slide_name.replace('.svs', '.db')
            image = read_ldmb(path, self.mapping_dict, img_name)
       
        if self.transform is not None:
            image = self.transform(image)

        return image



# function to train model and display accuracy, loss for train and val set   
# code adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def train_attn(model, # attention model
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          transforms_train,
          transforms_valid,
          mapping_dict,
          db_path,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=1,
          scheduler=None,
          num_workers=4):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
        logging.info(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')
        logging.info(f'Starting Training from Scratch.\n')

    overall_start = timer()

    if 'RESNET18' in save_file_name:
        # model for extracting tile features
        feature_model = ResNet_extractor(layers=18).cuda()
    elif 'RESNET50' in save_file_name:
        feature_model = ResNet_extractor(layers=50).cuda()
    else:
        print('Wrong model name!')
        exit()

    feature_model = feature_model.eval()
    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()
        
        # Decay Learning Rate
        if (scheduler is not None) and (epoch > 0):
            scheduler.step()

        for ii, (img_paths_list, targets) in enumerate(train_loader):

            targets = targets.cuda()
            
            # get tile features
            features_lists = []
            for img_paths in img_paths_list:
                dataset = ImageDataset(img_paths, transforms_train, db_path=db_path, mapping_dict=mapping_dict)
                dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)

                features_list = []

                for jj, (tiles) in enumerate(dataloader):
                    with torch.no_grad():
                        tiles = tiles.cuda()
                        features = feature_model(tiles)
                        features_list += features
                    del tiles
                    torch.cuda.empty_cache()

                features_lists.append(torch.stack(features_list))

            # pad features_lists such that we can stack and input in model in parallel
            features_lists_padded = pad_sequence(features_lists, batch_first=True, padding_value=0).cuda()

            # create attention mask to ignore padding in gated attention model
            L = torch.Tensor([f.shape[0] for f in features_lists])
            max_length = max(L)
            boolean_mask = L.unsqueeze(1)  > torch.arange(max_length)
            attention_mask = torch.zeros_like(boolean_mask, dtype=torch.float32)
            attention_mask[~boolean_mask] = float("-inf")
            attention_mask = attention_mask.cuda()

            # attention on top of tile features
            output, _ = model(features_lists_padded, attention_mask)

            # get loss
            output = output.squeeze()
            targets = targets.float() 
            if len(img_paths_list)==1:
                output = output.unsqueeze(0) 

            loss = criterion(output, targets)

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            size = targets.size(0)
            train_loss += loss.item() #* size
            pred = torch.round(torch.sigmoid(output)) 
            correct_tensor = pred.eq(targets.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc += accuracy.item() #* size

            if (ii % 2 == 0):
                # Track training progress
                print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')
            
        # After training loops ends, start validation
        model.epochs += 1

        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            pred_probs = []
            pred_targets = []

            for ii, (img_paths_list, targets) in enumerate(valid_loader):

                targets = targets.cuda()

                # get tile features
                features_lists = []

                for img_paths in img_paths_list:
                    dataset = ImageDataset(img_paths, transforms_valid, db_path=db_path, mapping_dict=mapping_dict)
                    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=3)

                    features_list = []

                    for jj, (tiles) in enumerate(dataloader):
                        with torch.no_grad():
                            tiles = tiles.cuda()
                            features = feature_model(tiles)
                            features_list += features
                        del tiles
                        torch.cuda.empty_cache()

                    features_lists.append(torch.stack(features_list))
                
                # pad features_lists such that we can stack and input in model in parallel
                features_lists_padded = pad_sequence(features_lists, batch_first=True, padding_value=0).cuda()
                
                # create attention mask to ignore padding in gated attention model
                L = torch.Tensor([f.shape[0] for f in features_lists])
                max_length = max(L)
                boolean_mask = L.unsqueeze(1)  > torch.arange(max_length)
                attention_mask = torch.zeros_like(boolean_mask, dtype=torch.float32)
                attention_mask[~boolean_mask] = float("-inf")
                attention_mask = attention_mask.cuda()

                # attention on top of tile features
                output, _ = model(features_lists_padded, attention_mask)

                # get loss
                output = output.squeeze()
                targets = targets.float()  
                if len(img_paths_list)==1:
                    output = output.unsqueeze(0) 
                loss = criterion(output, targets)

                # track metrics
                size = targets.size(0)
                valid_loss += loss.item() #* size
                prob = torch.sigmoid(output)
                pred = torch.round(torch.sigmoid(output))
                correct_tensor = pred.eq(targets.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                valid_acc += accuracy.item() #* size

                pred_probs += prob.cpu().numpy().flatten().tolist()
                pred_targets += targets.cpu().numpy().flatten().tolist()

            # Calculate average losses
            train_loss = train_loss / len(train_loader) #.dataset)
            valid_loss = valid_loss / len(valid_loader) #.dataset)

            # Calculate average accuracy
            train_acc = train_acc / len(train_loader) #.dataset)
            valid_acc = valid_acc / len(valid_loader) #.dataset)

            # Calculate auc
            valid_auc = roc_auc_score(pred_targets, pred_probs)
            
            history.append([train_loss, valid_loss, train_acc, valid_acc, valid_auc])

            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')
                print(f'\t\tValidation AUC: {valid_auc:.4f}')
                logging.info(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                logging.info(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')
                logging.info(f'\t\tValidation AUC: {valid_auc:.4f}')

            # Save the model if validation loss decreases
            if valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                state = {
                        'epoch': epoch,
                        'optimizer': optimizer.state_dict()    
                    }
                if scheduler != None:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, save_file_name.replace('-.pt','-state.pt')) # save optimizer and nr epochs
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch

            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
                    total_time = timer() - overall_start
                    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')
                    logging.info(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
                    logging.info(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')

                    # Load the best state dict
                    model.load_state_dict(torch.load(save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer

                    # Format history
                    history = pd.DataFrame(
                        history,
                        columns=[
                            'train_loss', 'valid_loss', 'train_acc',
                            'valid_acc', 'valid_auc'
                        ])
                    return model, history
    
    # Load the best state dict
    model.load_state_dict(torch.load(save_file_name))
    
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%')
    logging.info(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%')
    
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'valid_auc'])

    return model, history




