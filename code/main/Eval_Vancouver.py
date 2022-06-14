# Import Libraries
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

import pandas as pd
import logging
import re

import argparse
import os

from loaders import default_loader
from eval_functions import get_tile_performances_vanc, get_eval_df_vanc

from types import SimpleNamespace  


def add_tilename(row):
    region_len = len(row['matching_id'])
    return row['tile_name'][region_len+1:]


class EvalTilesDataset(Dataset):
    def __init__(self, df, transform=None):

        self.transform = transform
        self.targets = df[n.gene].values
        self.img_paths = df['tile_path'].values
        self.patient_ids = df['patient_id'].values
        self.matching_ids = df['matching_id'].values
        self.filenames = df['filename'].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        
        image = default_loader(self.img_paths[index])
        label = self.targets[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        patient_id = self.patient_ids[index]
        tile_path = self.img_paths[index]
        matching_id = self.matching_ids[index]
        filename = self.filenames[index]

        return image, label, patient_id, tile_path, matching_id, filename

if __name__=='__main__':
    # set up arg parser for command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest-folder', type=str, required=True)
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--gpu-num', type=str, required=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    foldername = args.dest_folder
    data_folder = args.data_folder

    model_run_folder= 'runs/TCGA_train/seed_273/'
    study = 'vanc'

    variables = {

        # general
        'source_folder':data_folder+'/TCGA_training/'+model_run_folder,
        'save_path':data_folder+'/TCGA_training/runs/'+study.upper()+'_eval/'+foldername+'/',
        'gene':'TP53',
        
        # model settings
        'model_name':"RESNET18",
        'panda':False,

        # preprocess settings
        'crop_size':224,
    }

    # define variables in namespace
    n = SimpleNamespace(**variables)
    
    blurred_configs = ['Vancouver_blurred_tile_paths_NEW.txt']
    penm_configs = ['Vancouver_penm_tile_paths_NEW.txt']
        
    # model training settings
    batch_size = 512

    # set up logger (such that it can be used in useful_functions.py as well)
    logger = logging.getLogger('vanc_logger')

    # config log file
    logging.basicConfig(filename=n.save_path+n.model_name+'-logfile-',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    # log variables
    for key, value in variables.items():  
        logging.info('%s:%s\n' % (key, value))

    # read dataframe
    test_df = pd.read_csv(data_folder+'/TCGA_training/vancouver_df.csv')
    
    # remap tile path
    path_to_tiles = data_folder+'/TCGA_tiles/Vancouver_tiles_NEW/'
    test_df['tile_path'] = test_df['tile_path'].apply(lambda x: re.sub('/home/bioit/mpizuric/Documents/0VancouverMaskGen/Vancouver_tiles_NEW/',path_to_tiles,x))
    print('Test df shape: '+str(test_df.shape))

    # set tile name
    test_df['tile_name'] = test_df['tile_path'].str.split('/').str[-1]

    # remove blurred tiles
    bad_tile_names_all = []
    for blurred_config in blurred_configs:
        text_file = open(data_folder+'/TCGA_training/'+blurred_config, "r")
        bad_tile_paths = text_file.read().split('\n')      
        bad_tile_names = [(i.split('/')[-1]) for i in bad_tile_paths]
        bad_tile_names_all += bad_tile_names

    test_df = test_df[~test_df['tile_name'].isin(bad_tile_names_all)]
    print('Shape after remove blurred: '+str(test_df.shape))

    # remove tiles with pen marks
    pen_mark_tile_names_all = []
    for penmark_config in penm_configs:
        text_file = open(data_folder+'/TCGA_training/'+penmark_config, "r")
        pen_mark_tile_paths = text_file.read().split('\n')      
        pen_mark_tile_names = [(i.split('/')[-1]) for i in pen_mark_tile_paths]
        pen_mark_tile_names_all += pen_mark_tile_names
    test_df = test_df[~test_df['tile_name'].isin(pen_mark_tile_names)]
    print('Shape after remove penm: '+str(test_df.shape))
    
    # Image transformations
    image_transforms = {
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize((n.crop_size, n.crop_size)),
            transforms.CenterCrop(size=n.crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # define dataset and dataloader
    test_dataset = EvalTilesDataset(test_df, transform=image_transforms['valid'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=3)

    # get models
    models = []
    for i in range(3):

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        model.load_state_dict(torch.load(n.source_folder+'RESNET18-TCGA-'+str(i)+'-.pt'))

        models.append(model)
        
    # eval models
    for i in range(3):
        final_auc, patient_ids_test, labels_test, probs_test, preds_test, tile_paths_test, matching_ids_test, filenames_test = get_tile_performances_vanc(models[i], test_dataloader)
        print("Tile level AUC for fold "+str(i)+": "+str(final_auc))
    
        test_eval_df =  get_eval_df_vanc(patient_ids_test, labels_test, probs_test, preds_test, tile_paths_test, matching_ids_test, filenames_test)
        print('Calculated test eval df')

        test_eval_df.to_csv(n.save_path+study+'_eval_'+str(i)+'.csv')
        print('Saved df')


        