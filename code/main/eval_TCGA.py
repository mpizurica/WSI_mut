# Import Libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import numpy as np
import pandas as pd

from types import SimpleNamespace  

import argparse
from distutils.util import strtobool

from loaders import default_loader
from eval_functions import get_eval_df, get_tile_performances
from preprocess import get_dataframe, read_ldmb


class EvalTilesDataset(Dataset):
    def __init__(self, df, transform=None, db_path=None, mapping_dict=None):

        self.transform = transform
        self.targets = df[n.gene].values
        self.img_paths = df['tile_path'].values
        self.patient_ids = df['patient_id'].values
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
        
        patient_id = self.patient_ids[index]
        tile_path = self.img_paths[index]

        return image, label, patient_id, tile_path


if __name__ == '__main__':

    # set up arg parser for command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest-folder', type=str, required=True)
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--model-folder', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--annot', type=lambda x: bool(strtobool(str(x))), required=True)
    parser.add_argument('--all-regions', type=lambda x: bool(strtobool(str(x))), required=True)
    parser.add_argument('--use-db', type=lambda x: bool(strtobool(str(x))), required=True)
    parser.add_argument('--gpu-num', type=str, required=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    foldername = args.dest_folder

    # folder where data is located and where results will be written to
    # project or project_scratch for GPUlab
    # bmir ct server: /labs/gevaertlab/data/prostate_cancer/
    data_folder = args.data_folder
    model_run_folder = args.model_folder

    variables = {
        'tiles_from_annot':args.annot, 
        'source':args.source,
        'all_regions':args.all_regions,

        # general
        'model_name':"RESNET18",
        'crop_size': 224,
        'gene':"TP53",

        # where to save results
        'save_path':data_folder+'TCGA_training/runs/TCGA_eval/'+foldername+'/',
        
        # which model to use
        'source_folder':data_folder+'TCGA_training/runs/TCGA_train/'+model_run_folder+'/',
    }
    
    # define variables in namespace
    n = SimpleNamespace(**variables)

    os.mkdir(n.save_path)

    # database where tiles are stored
    if args.use_db:
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

    # Image transformations
    image_transforms = {
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize((n.crop_size, n.crop_size)),
            transforms.CenterCrop(size=n.crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # get dataframe name
    suffixes = {'TP53':'', 'all_regions':'_all_regions','all_tiles':'_all_tiles'}
    complete_train_df =  get_dataframe(data_folder=data_folder, suffixes=suffixes, gene=n.gene, \
                                        tiles_from_annot=n.tiles_from_annot, all_regions=n.all_regions, \
                                        attention=False, train_or_test='TRAIN')
    test_df = get_dataframe(data_folder=data_folder, suffixes=suffixes, gene=n.gene, \
                                        tiles_from_annot=n.tiles_from_annot, all_regions=n.all_regions, \
                                        attention=False, train_or_test='TEST')

    # get val dfs and test df
    val_dfs = []
    for i in range(3):
        val_dfs.append(complete_train_df[complete_train_df['patient_id'].isin(np.load(n.source_folder+'val'+str(i)+'.npy', allow_pickle=True))])

    # define dataloaders
    batch_size = 512
    val_dataloaders = []
    test_dataloaders = []
    for i in range(3):
        
        val_dataset = EvalTilesDataset(val_dfs[i], transform=image_transforms['valid'], db_path=db_path, mapping_dict=mapping_dict)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=10)
        val_dataloaders.append(val_dataloader)

    test_dataset = EvalTilesDataset(test_df, transform=image_transforms['valid'], db_path=db_path, mapping_dict=mapping_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=10)

    # evaluate model
    all_models = []
    for i in range(3):

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        model.load_state_dict(torch.load(n.source_folder+'RESNET18-TCGA-'+str(i)+'-.pt'))

        model.eval()
        model = model.to('cuda')
        all_models.append(model)

    # we should evaluate validation set on model that was trained on corresponding training set
    # model '0' was trained on the training set which is the equivalent of 'validation set 1 and 2' 
    # in other words, we should evaluate model[0] on val_df[0] 
    # this is different for the test set, on which we can evaluate all models
    # tile_perfs is of the shape final_auc, patient_ids, labels, probs, preds, tile_paths
    tile_perfs_val = []
    tile_perfs_test = []
    for i in range(3):
        print('Evaluating fold '+str(i))
        tile_perfs_val.append(get_tile_performances(all_models[i], val_dataloaders[i]))
        tile_perfs_test.append(get_tile_performances(all_models[i], test_dataloader))

    # print AUCs for validation and test
    print('Validation set: '+str(tile_perfs_val[0][0])+', '+str( tile_perfs_val[1][0])+', '+str( tile_perfs_val[2][0]))
    print('Test set: '+str(tile_perfs_test[0][0])+', '+str(tile_perfs_test[1][0])+', '+str(tile_perfs_test[2][0]))

    # create dataframe for tile performances
    val_eval_dfs = []
    test_eval_dfs = []
    for i in range(3):
        val_eval_df = get_eval_df(tile_perfs_val[i][1], tile_perfs_val[i][2], tile_perfs_val[i][3], tile_perfs_val[i][4], tile_perfs_val[i][5])
        test_eval_df = get_eval_df(tile_perfs_test[i][1], tile_perfs_test[i][2], tile_perfs_test[i][3], tile_perfs_test[i][4], tile_perfs_test[i][5])
        val_eval_dfs.append(val_eval_df)
        test_eval_dfs.append(test_eval_df)

        val_eval_df.to_csv(n.save_path+'val_eval_'+str(i)+'.csv')
        test_eval_df.to_csv(n.save_path+'test_eval_'+str(i)+'.csv')

    print('calculated eval dfs')

