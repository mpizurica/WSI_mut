# Import Libraries
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import numpy as np
import pandas as pd
import logging
import datetime
import random

import argparse
from distutils.util import strtobool

from types import SimpleNamespace  

from train_functions import train, train_attn
from preprocess import read_ldmb, get_dataframe, stratified_group_k_fold, get_undersampled_df, \
                       get_undersampled_dfs, undersample_df_nr_tiles_pp_random
from loaders import default_loader
from define_model import get_final_model

from sklearn.model_selection import StratifiedGroupKFold


class TilesDataset(Dataset):
    """
    custom dataset class to load images with
    corresponding df that contains labels
    df must contain image path, label
    """
    def __init__(self, df, transform=None, db_path=None, mapping_dict=None):
       
        self.transform = transform
        self.targets = df[n.gene].values
        self.img_paths = df['tile_path'].values
        self.db_path = db_path
        self.mapping_dict = mapping_dict

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        if db_path == None:
            image = default_loader(img_path)
        else:
            img_name = img_path.split('/')[-1]
            slide_name = img_name.split('_')[0]
            path = self.db_path+'/'+slide_name.replace('.svs', '.db')
            image = read_ldmb(path, self.mapping_dict, img_name)
        
        label = self.targets[index]
       
        if self.transform is not None:
            image = self.transform(image)

        return image, label

class SlidesDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        self.grouped = df.groupby('slide_name').head(1)
        self.slide_names = self.grouped['slide_name'].values
        self.targets = self.grouped[n.gene].values

    def __len__(self):
        return len(self.slide_names)

    def __getitem__(self, index):
        slide = self.slide_names[index]
        slide_df = self.df[self.df['slide_name']==slide]
        
        label = slide_df.iloc[0][n.gene]
        img_paths = slide_df['tile_path'].values

        return img_paths, label


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

if __name__ == '__main__':

    # set up arg parser for command line
    parser = argparse.ArgumentParser()

    # setup arguments
    parser.add_argument('--dest-folder', type=str, required=True)
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--gpu-num', type=str, required=True)
    parser.add_argument('--use-db', type=lambda x: bool(strtobool(str(x))), required=True)
    parser.add_argument('--num-workers', type=str, required=True)
    parser.add_argument('--train-folds', type=str, required=True) #which folds to train, separated by comma

    # learning arguments
    parser.add_argument('--seed-crossval', type=str, required=True)
    parser.add_argument('--indices-path', type=str, required=True)
    parser.add_argument('--batch-size', type=str, required=True)
    parser.add_argument('--lr', type=str, required=True)
    parser.add_argument('--num-epochs', type=str, required=True)
    parser.add_argument('--use-scheduler', type=str, required=True)

    # annotation and attention arguments
    parser.add_argument('--all-regions', type=lambda x: bool(strtobool(str(x))), required=True)
    parser.add_argument('--annot', type=lambda x: bool(strtobool(str(x))), required=True)
    parser.add_argument('--attention', type=lambda x: bool(strtobool(str(x))), required=True)

    # undersample settings
    parser.add_argument('--multiple-undersamplings', type=lambda x: bool(strtobool(str(x))), required=True)
    parser.add_argument('--undersample-validation', type=lambda x: bool(strtobool(str(x))), required=True)
    parser.add_argument('--undersample-train', type=lambda x: bool(strtobool(str(x))), required=True) # we always undersample train in tile level model, this argument is for attention based model

    args = parser.parse_args()

    # get foldername for destination folder from shell script 
    foldername = args.dest_folder

    # folder where data is located and where results will be written to
    # project or project_scratch for GPUlab
    # bmir ct server: /labs/gevaertlab/data/prostate_cancer/
    data_folder = args.data_folder

    # folds to train
    train_folds = args.train_folds

    if foldername == 'None':
        foldername = ''
        if args.attention: 
            foldername += 'attn_'
        if args.annot == False:
            foldername += 'allt_'
        elif args.all_regions:
            foldername += 'allr_'
        foldername += 'seed_'+args.seed_crossval
        if (args.annot == True) and (args.undersample_train == True) and (args.attention == True):
            foldername += '_undersample'
        
    # create dest folder
    save_dir =  data_folder + '/TCGA_training/runs/TCGA_train/' + foldername
    os.mkdir(save_dir)

    # general settings
    use_db = args.use_db
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num 

    variables = {
        # whether to use tiles only from annotated regions
        'tiles_from_annot':args.annot,
        'all_regions': args.all_regions,
        'num_folds':3,

        # attention params
        'attention':args.attention,
        'attention_module':'faisal', 
        'hidden_dim':256, 
        'hidden_dim2':128, 
        'dropout':0.25, #if None then no dropout, orig: 0.25
        'use_b':True,

        # preprocessing settings
        'multiple_undersamplings':args.multiple_undersamplings,
        'undersample_validation':args.undersample_validation,
        'undersample_train':args.undersample_train,
        'seed_crossval': int(args.seed_crossval),

        # augmentation settings
        'brightness':0.2, 
        'contrast':0.2, 
        'saturation':0.1,
        'hue':0.1,

        # set data settings
        'gene':'TP53',
        'size_pixels':512,
        'crop_size':224,
        'indices_path': args.indices_path,

        # model settings
        'model_name':"RESNET18",
        'panda':False,

        # training settings
        'batch_size':int(args.batch_size),
        'learning_rate':float(args.lr), 
        'weight_decay':0,
        'num_epochs':int(args.num_epochs),
        'max_epochs_stop':150,
        'use_scheduler':args.use_scheduler, 
        'pin_memory':False,
        'num_workers':int(args.num_workers),
        
        'save_path':data_folder+'TCGA_training/runs/TCGA_train/'+foldername+'/'
    }

    # define variables in namespace
    n = SimpleNamespace(**variables)

    if n.attention == False:
        n.attention_module = None

    # set amount tiles per patient
    if n.attention == False: # in this case, we will later also undersample for class balance
        if n.tiles_from_annot:
            amt_tiles_pp = 500 # will be smaller after undersampling for class balance
        else:
            amt_tiles_pp = 50 # less here, but will turn out equivalent after undersampling for class balance 
                              #(more tiles from minority class are present because we take tiles from the whole slide now, so we will need to undersample less later)

    else: # we don't undersample tiles anymore later
        amt_tiles_pp = 50

    # set seeds for reproducability
    np.random.seed = n.seed_crossval
    torch.manual_seed(0)
    random.seed(0)

    # set datetime for logging
    current_time = datetime.datetime.now()

    # set up logger (such that it can be used in useful_functions.py as well)
    logger = logging.getLogger('tcga_logger')
    logging.basicConfig(filename=n.save_path+n.model_name+'-logfile-'+str(current_time),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    # log config
    for key, value in variables.items():  
        logging.info('%s:%s' % (key, value))

    # get dataframe name
    suffixes = {'LN':'LN', 'TP53':'', 'all_regions':'_all_regions','all_tiles':'_all_tiles', 'attention': '_attention'}
    complete_train_df =  get_dataframe(data_folder, suffixes, n.gene, n.tiles_from_annot, \
                                        n.all_regions, n.attention, 'TRAIN')

    # remove blurred and pen marked
    print('Before removing blurred and penm: '+str(complete_train_df.shape[0]))
    complete_train_df['tile_name'] = complete_train_df['tile_path'].str.split('/').str[-1]

    # database where tiles are stored
    if use_db:
        db_path = data_folder+'TCGA_tiles/'
        if n.tiles_from_annot:
            suffix = ''
            db_path += 'db_tiles_512px'
        else:
            suffix = '_all_tiles'
            db_path += 'db_all_tiles_512px'
        mapping_df = pd.read_csv('~/code/WSI_mutation/code/img_name_to_index'+suffix+'.csv', index_col=0).set_index('img_name')
        mapping_dict = mapping_df.to_dict()['index']
    else:
        db_path = None
        mapping_dict = None

    # blurred was already removed upfront
    # text_file = open(data_folder+"TCGA_training/blurred_tile_paths.txt", "r")
    # blurred_tile_paths = text_file.read().split('\n') 
    # blurred_tile_names = [i.split('/')[-1] for i in blurred_tile_paths]
    # complete_train_df = complete_train_df[~complete_train_df['tile_name'].isin(blurred_tile_names)]

    # remove pen marked
    text_file = open(data_folder+"TCGA_training/penm_tile_paths_new.txt", "r")
    penm_tile_paths = text_file.read().split('\n') 
    penm_tile_names = [i.split('/')[-1] for i in penm_tile_paths]
    complete_train_df = complete_train_df[~complete_train_df['tile_name'].isin(penm_tile_names)]
    print('After removing penm: '+str(complete_train_df.shape[0]))

    # shuffle train dataframe
    final_df = complete_train_df.sample(frac=1, random_state=n.seed_crossval).reset_index(drop=True)
    train_dfs = []
    val_dfs = []

    if n.indices_path != 'None':
        print('Loading indices from path')
        logging.info('Loading indices from path, %s' %(n.indices_path))
        for i in range(n.num_folds):
            train_ind = np.load(n.indices_path+'train'+str(i)+'.npy', allow_pickle=True)
            val_ind = np.load(n.indices_path+'val'+str(i)+'.npy', allow_pickle=True)
            train_dfs.append(complete_train_df[complete_train_df['patient_id'].isin(train_ind)])
            val_dfs.append(complete_train_df[complete_train_df['patient_id'].isin(val_ind)])

    else:
        if n.attention == False:

            # cross-validation split
            X = final_df['tile_path'].values
            y = final_df[n.gene].values
            groups = final_df['patient_id'].values
            cv = StratifiedGroupKFold(n_splits=n.num_folds,shuffle=True,random_state=n.seed_crossval)

            for train_idxs, test_idxs in cv.split(X, y, groups):
                train_dfs.append(final_df[final_df['patient_id'].isin(groups[train_idxs])])
                val_dfs.append(final_df[final_df['patient_id'].isin(groups[test_idxs])])

        else: # split happens on slide-level instead of on tile-level
            temp = final_df.groupby(final_df['slide_name'])
            temp = temp.head(1)[['slide_name',n.gene,'patient_id']]

            X = temp.index.values
            y = temp[n.gene].values
            groups = temp["patient_id"].values
            cv = StratifiedGroupKFold(n_splits=n.num_folds,shuffle=True,random_state=n.seed_crossval)

            for train_idxs, test_idxs in cv.split(X, y, groups):
                train_dfs.append(final_df[final_df['patient_id'].isin(groups[train_idxs])])
                val_dfs.append(final_df[final_df['patient_id'].isin(groups[test_idxs])])

    # reset indices
    for i in range(n.num_folds):
        train_dfs[i] = train_dfs[i].reset_index(drop=True)
        val_dfs[i] = val_dfs[i].reset_index(drop=True)

    # write patient indices
    for i in range(n.num_folds):
        np.save(n.save_path+'val'+str(i)+'.npy', np.unique(val_dfs[i]['patient_id'].values))
        np.save(n.save_path+'train'+str(i)+'.npy', np.unique(train_dfs[i]['patient_id'].values))

    logging.info('saved indices for train val set')
    
    # undersampling dataframes if wanted
    # if attention is used, we do not undersample, except for the case when the whole WSI is used
    if n.attention == False:
        if n.multiple_undersamplings == False:
            for i in range(n.num_folds):
                train_dfs[i] = get_undersampled_df(train_dfs[i], amt_tiles_pp, n.gene, n.tiles_from_annot)

                if ((n.undersample_validation == True) and (n.tiles_from_annot == False)):
                    # if using all tiles, validation dataframe is too large --> eval on subset
                    # not to be used in other configurations (dominant tumor regions, all tumor regions)
                    val_dfs[i] = val_dfs[i].sample(n=10000, random_state=n.seed_crossval).reset_index()

            print('amount of train samples: '+str(train_dfs[0].shape[0])+', '+str(train_dfs[1].shape[0])+', '+str(train_dfs[2].shape[0]))
            print('amount of train samples, mut: '+str(train_dfs[0][train_dfs[0][n.gene]==1].shape[0])+', '+str(train_dfs[1][train_dfs[1][n.gene]==1].shape[0])+', '+str(train_dfs[2][train_dfs[2][n.gene]==1].shape[0]))
            print('amount of train samples, no mut: '+str(train_dfs[0][train_dfs[0][n.gene]==0].shape[0])+', '+str(train_dfs[1][train_dfs[1][n.gene]==0].shape[0])+', '+str(train_dfs[2][train_dfs[2][n.gene]==0].shape[0]))
            
            print('amount of val samples: '+str(val_dfs[0].shape[0])+', '+str(val_dfs[1].shape[0])+', '+str(val_dfs[2].shape[0]))

        else:
            # undersample multiple times (if undersampling is very heavy, we are using only a small part of the data; \
            # this allows to use different undersampled train sets every epoch)
            train_ind_epoch = get_undersampled_dfs(train_dfs, n.tiles_from_annot, amt_tiles_pp, n.gene)

            print('amount of train samples: '+str(len(train_ind_epoch[0][0]))+', '+str(len(train_ind_epoch[1][0]))+', '+str(len(train_ind_epoch[2][0])))

            # if using all tiles, validation dataframe is too large --> eval on subset
            if ((n.undersample_validation == True) and (n.tiles_from_annot == False)):
                val_ind_epoch = [[],[],[]]
                for i in range(n.num_folds):
                    for j in range(30):
                        val_ind_epoch[i].append(val_dfs[i].sample(n=10000, random_state=j).index.values)

                print('amount of val samples: '+str(len(val_ind_epoch[0][0]))+', '+str(len(val_ind_epoch[1][0]))+', '+str(len(val_ind_epoch[2][0])))

        logging.info('done with undersampling')
        print('done with undersampling')

    else:
        if ((n.undersample_train == True) or (n.undersample_validation == True)):
            if n.multiple_undersamplings == False:
                for i in range(n.num_folds):
                    if n.undersample_train:
                        train_dfs[i] = undersample_df_nr_tiles_pp_random(train_dfs[i], amount=amt_tiles_pp, \
                                                                        random_state=0).sample(frac = 1).reset_index(drop=True)
                    if n.undersample_validation:
                        val_dfs[i] = undersample_df_nr_tiles_pp_random(val_dfs[i], amount=amt_tiles_pp, \
                                                                    random_state=0).sample(frac = 1).reset_index(drop=True)
                print('done with undersampling')

                print('amount of train samples: '+str(train_dfs[0].shape[0])+', '+str(train_dfs[1].shape[0])+', '+str(train_dfs[2].shape[0]))
                print('amount of train samples, mut: '+str(train_dfs[0][train_dfs[0][n.gene]==1].shape[0])+', '+str(train_dfs[1][train_dfs[1][n.gene]==1].shape[0])+', '+str(train_dfs[2][train_dfs[2][n.gene]==1].shape[0]))
                print('amount of train samples, no mut: '+str(train_dfs[0][train_dfs[0][n.gene]==0].shape[0])+', '+str(train_dfs[1][train_dfs[1][n.gene]==0].shape[0])+', '+str(train_dfs[2][train_dfs[2][n.gene]==0].shape[0]))
                
                print('amount of val samples: '+str(val_dfs[0].shape[0])+', '+str(val_dfs[1].shape[0])+', '+str(val_dfs[2].shape[0]))

            else:
                print('Multiple undersamplings not implemented for attention')
                exit()

    # Image transformations
    image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.Resize((n.crop_size, n.crop_size)),
            transforms.ColorJitter(brightness=n.brightness, contrast=n.contrast, saturation=n.saturation, hue=n.hue),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(size=n.crop_size), # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]) # Imagenet standards
        ]),
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize((n.crop_size, n.crop_size)),
            transforms.CenterCrop(size=n.crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Datasets
    train_keys = ['train_'+str(i) for i in range(n.num_folds)]
    val_keys = ['val_'+str(i) for i in range(n.num_folds)]
    data = {}

    if n.attention:
        for i, key in enumerate(train_keys):
            data[key] = SlidesDataset(df=train_dfs[i], transform=image_transforms['train'])
        for i, key in enumerate(val_keys):
            data[key] = SlidesDataset(df=val_dfs[i], transform=image_transforms['valid'])

    else:
        for i, key in enumerate(train_keys):
            data[key] = TilesDataset(df=train_dfs[i], transform=image_transforms['train'], db_path=db_path, mapping_dict=mapping_dict)
        for i, key in enumerate(val_keys):
            data[key] = TilesDataset(df=val_dfs[i], transform=image_transforms['valid'], db_path=db_path, mapping_dict=mapping_dict)
    
    batch_size=n.batch_size

    # Dataloaders
    dataloaders = {}

    # Dataloaders tile level model
    if n.attention == False:
        if n.multiple_undersamplings == True:
            
            if n.undersample_validation == False:
                for i, key in enumerate(val_keys):
                    dataloaders[key] = DataLoader(data[key], batch_size=batch_size, shuffle=True,num_workers=n.num_workers)

            else:
                # if use all tiles --> validation set is also subsampled
                # do it multiple times 
                dataloaders_val_sets = [[],[],[]]
                for i in range(n.num_folds):
                    for j in range(len(val_ind_epoch[i])):
                        valset_i_j = torch.utils.data.Subset(data['val_'+str(i)], val_ind_epoch[i][j])
                        dataloader_i_j = DataLoader(valset_i_j, batch_size=batch_size, shuffle=True,num_workers=n.num_workers)
                        dataloaders_val_sets[i].append(dataloader_i_j)

            # train set is subsampled multiple times
            dataloaders_train_sets = [[],[],[]]
            for i in range(n.num_folds):
                for j in range(len(train_ind_epoch[i])):
                    trainset_i_j = torch.utils.data.Subset(data['train_'+str(i)], train_ind_epoch[i][j])
                    dataloader_i_j = DataLoader(trainset_i_j, batch_size=batch_size, shuffle=True,num_workers=n.num_workers)
                    dataloaders_train_sets[i].append(dataloader_i_j)

        else:
            # Dataloader iterators
            for i, key in enumerate(val_keys):
                dataloaders[key] = DataLoader(data[key], batch_size=batch_size, shuffle=True,num_workers=n.num_workers)
            for i, key in enumerate(train_keys):
                dataloaders[key] = DataLoader(data[key], batch_size=batch_size, shuffle=True,num_workers=n.num_workers)

    # dataloaders attention based model
    else:
        class_weights_all = []
        train_keys = [k for k in data.keys() if 'train' in k]
        train_keys.sort() # not really necessary because list stays in same order
        for i, dataset in enumerate( [data[key] for key in train_keys] ):
            slides = np.unique(train_dfs[i]['slide_name'].values)
            class_count = [0, 0]

            for slide in slides:
                slide_df = train_dfs[i][train_dfs[i]['slide_name']==slide]
                label = slide_df.iloc[0][n.gene]
                class_count[int(label)] += 1

            class_weights = 1./torch.tensor(class_count, dtype=torch.float)

            target_list = torch.LongTensor(dataset.targets) 
            class_weights_dataset = class_weights[target_list]
            class_weights_all.append(class_weights_dataset)

        weighted_samplers = []
        for i in range(n.num_folds):
            sampler  = WeightedRandomSampler(weights=class_weights_all[i],num_samples=len(class_weights_all[i]),
                replacement=True
            )
            weighted_samplers.append(sampler)

        for i, key in enumerate(train_keys):
            dataloaders[key] = DataLoader(data[key], batch_size=n.batch_size, collate_fn=my_collate, sampler=weighted_samplers[i], shuffle=False,num_workers=n.num_workers)
        for i, key in enumerate(val_keys):
            dataloaders[key] = DataLoader(data[key], batch_size=64, collate_fn=my_collate, num_workers=n.num_workers)

    # training model

    histories = []
    saved_models = []

    logging.info('initialized dataloaders')
    print('initialized dataloaders')

    for i in [int(j) for j in train_folds.split(",")]:
    #for i in range(n.num_folds):

        # get model
        model_i = get_final_model(n.model_name, n.attention_module, n.hidden_dim, n.hidden_dim2, \
                                    n.use_b, n.dropout)
        
        # specify params to update
        params_to_update = []
        for name, param in model_i.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                
        # specify loss function      
        criterion = nn.BCEWithLogitsLoss()
        
        # optimizer
        optimizer = optim.Adam(params_to_update, weight_decay = n.weight_decay, lr=n.learning_rate) 

        # initialize dataloaders
        val_loader = 'val_' + str(i)
        train_loader = 'train_' + str(i)

        # different dataloaders depending on settings
        if n.multiple_undersamplings == True:
            dataloader_train = dataloaders_train_sets[i]

            if n.undersample_validation:
                dataloader_val = dataloaders_val_sets[i]
            else:
                dataloader_val = dataloaders[val_loader]

        else:
            dataloader_train = dataloaders[train_loader]
            dataloader_val = dataloaders[val_loader]
        
        if n.use_scheduler == 'cosine':
            updates_per_ep = int(len(dataloader_train.dataset)/n.batch_size)
            steps = 10*updates_per_ep
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
            #scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=2, after_scheduler=scheduler_)
            print('using cosine scheduler')

        elif n.use_scheduler == 'step':
            # gamma = decaying factor
            # step_size: at how many multiples of epoch you decay
            steps = 10
            scheduler = StepLR(optimizer, step_size=steps, gamma=0.5) # new_lr = lr*gamma 
            print('using step scheduler')

        else:
            scheduler = None
            steps = None

        if n.attention:
            model_i, history = train_attn(
                model=model_i,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=dataloader_train,
                valid_loader=dataloader_val,
                transforms_train=image_transforms['train'],
                transforms_valid=image_transforms['valid'],
                mapping_dict=mapping_dict,
                db_path=db_path,
                save_file_name=n.save_path+n.model_name+'-'+'TCGA'+'-'+str(i)+'-.pt',
                max_epochs_stop=n.max_epochs_stop,
                n_epochs=n.num_epochs, 
                print_every=1,
                scheduler=scheduler,
                num_workers=n.num_workers)

        else:
            model_i, history = train(
                model=model_i,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=dataloader_train,
                valid_loader=dataloader_val,
                save_file_name=n.save_path+n.model_name+'-'+'TCGA'+'-'+str(i)+'-.pt',
                max_epochs_stop=n.max_epochs_stop,
                n_epochs=n.num_epochs, 
                print_every=1,
                scheduler=scheduler)
        
        np.save(n.save_path+'history-'+str(i)+'.npy', history)

    print("Saved models and histories")
        
