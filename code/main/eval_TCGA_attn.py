import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import os
import argparse
from distutils.util import strtobool

from attention_model import Attn_Net_Gated, ResNet_extractor
from loaders import default_loader
from preprocess import get_dataframe, read_ldmb


class SlidesDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        self.grouped = df.groupby('slide_name').head(1)
        self.slide_names = self.grouped['slide_name'].values
        self.targets = self.grouped[gene].values

    def __len__(self):
        return len(self.slide_names)

    def __getitem__(self, index):
        slide = self.slide_names[index]
        slide_df = self.df[self.df['slide_name']==slide]
        
        label = slide_df.iloc[0][gene]
        img_paths = slide_df['tile_path'].values

        return img_paths, label, slide


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


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.IntTensor(target) #torch.LongTensor(target)
    slide = [item[2] for item in batch]
    return [data, target, slide]


def get_performances_attn(feature_model, model, dataloader, transform, mapping_dict, db_path):
    """
    returns dataframe with slide level probability and attention weight for all tiles of slide
    resulting df has columns ['slide_name', 'label', 'prob', 'attn_w', 'tile_names']
    """
    labels = {} #slide_name:label
    probs = {} #slide_name:prob
    attention_weights = {} #slide_name:attention_weights
    tile_names = {} #slide_name:tile_names

    with torch.no_grad():
        model.eval()

        for ii, (img_paths_list, targets, slide_names) in enumerate(dataloader):

            targets = targets.cuda()

            # get tile features
            features_lists = []
            for ind, img_paths in enumerate(img_paths_list):
                dataset = ImageDataset(tile_paths_list=img_paths, transform=transform, \
                                        db_path=db_path, mapping_dict=mapping_dict)
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

                tile_names[slide_names[ind]] = ','.join(str(x) for x in img_paths) #convert to string to later be able to put in df
            
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
            output, attn = model(features_lists_padded, attention_mask)
            
            attn = attn.cpu().detach().numpy()
            prob = list(torch.sigmoid(output).cpu().detach().numpy().flatten())
            targets = list(targets.cpu().detach().numpy().flatten())
            
            for s, slide_name in enumerate(slide_names):
                if sum(attn[s][0,len(tile_names[slide_names[s]]):]) != 0:
                    print('padding went wrong!!!')
                    exit()
                attn_w = attn[s][0,:len(tile_names[slide_names[s]])] # remove padding
                attention_weights[slide_name] = ','.join(str(x) for x in attn_w) #convert to string to later be able to put in df
                probs[slide_name] = prob[s]
                labels[slide_name] = targets[s]

    df = pd.DataFrame([labels, probs, attention_weights, tile_names]).T.reset_index()
    df.columns = ['slide_name', 'label', 'prob', 'attn_w', 'tile_names']
    
    return df

if __name__ == '__main__':
    # set up arg parser for command line
    parser = argparse.ArgumentParser()

    # setup arguments
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--all-regions', type=lambda x: bool(strtobool(str(x))), required=True)

    args = parser.parse_args()

    # setup variables
    gene = 'TP53'
    all_regions = args.all_regions
    use_db = True
    num_workers = 4

    # model variables
    model_name = 'RESNET18'
    run_name = args.run_name 

    # attention variables
    hidden_dim = 256 
    hidden_dim2 = 128 
    dropout = True
    use_b = True

    if model_name == 'RESNET18':
        dim = 512
    elif model_name == 'RESNET50':
        dim = 1024

    # data and path variables
    data_folder = '/oak/stanford/groups/ogevaert/data/Prad-TCGA/' #'/labs/gevaertlab/data/prostate_cancer/'
    run_folder = 'TCGA_training/runs/TCGA_train/'+run_name+'/'
    path = data_folder+run_folder
    save_path = data_folder+'TCGA_training/runs/TCGA_eval/'+run_name+'/'
    os.mkdir(save_path)

    if use_db:
        db_path = data_folder+'TCGA_tiles/db_tiles_512px/'
        mapping_df = pd.read_csv('~/code/WSI_mutation/code/img_name_to_index.csv', index_col=0).set_index('img_name')
        mapping_dict = mapping_df.to_dict()['index']
    else:
        db_path = None
        mapping_dict = None

    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # get dataframe 
    suffixes = {'LN':'LN', 'TP53':'', 'all_regions':'_all_regions','all_tiles':'_all_tiles', 'attention':'_attention'}
    complete_train_df =  get_dataframe(data_folder=data_folder, labels='kather_tcga', suffixes=suffixes, gene='TP53', \
                                        tiles_from_annot=True, all_regions=all_regions, attention=True, train_or_test='TRAIN')
    test_df  = get_dataframe(data_folder=data_folder, labels='kather_tcga', suffixes=suffixes, gene='TP53', \
                                        tiles_from_annot=True, all_regions=all_regions, attention=True, train_or_test='TEST')

    # remove blurred and pen marked
    complete_train_df['tile_name'] = complete_train_df['tile_path'].str.split('/').str[-1]
    test_df['tile_name'] = test_df['tile_path'].str.split('/').str[-1]

    text_file = open(data_folder+"TCGA_training/penm_tile_paths_new.txt", "r")
    penm_tile_paths = text_file.read().split('\n') 
    penm_tile_names = [i.split('/')[-1] for i in penm_tile_paths]

    complete_train_df = complete_train_df[~complete_train_df['tile_name'].isin(penm_tile_names)]
    test_df = test_df[~test_df['tile_name'].isin(penm_tile_names)]

    # load val datasets
    val_dfs = []
    print('Loading patients from path')
    for i in range(3):
        val_ind = np.load(path+'val'+str(i)+'.npy', allow_pickle=True)
        val_dfs.append(complete_train_df[complete_train_df['patient_id'].isin(val_ind)].reset_index(drop=True))

    # Image transformations
    image_transforms = {
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Datasets
    data = {
            'val_0':SlidesDataset(df=val_dfs[0], transform=image_transforms['valid']),
            'val_1':SlidesDataset(df=val_dfs[1], transform=image_transforms['valid']),
            'val_2':SlidesDataset(df=val_dfs[2], transform=image_transforms['valid']),
            'test':SlidesDataset(df=test_df, transform=image_transforms['valid']),
    }

    # Dataloaders
    dataloaders = {#shuffle has to be false for the weighted random sampler (the sampler shuffles the data because it samples randomly)
            'val_0': DataLoader(data['val_0'], batch_size=32, collate_fn=my_collate, num_workers=num_workers), 
            'val_1': DataLoader(data['val_1'], batch_size=32, collate_fn=my_collate, num_workers=num_workers),
            'val_2': DataLoader(data['val_2'], batch_size=32, collate_fn=my_collate, num_workers=num_workers),
            'test': DataLoader(data['test'], batch_size=32, collate_fn=my_collate, num_workers=num_workers),
    }

    # model for extracting tile features
    import re
    num_layers = int(re.findall(r'\d+', model_name)[0])
    feature_model = ResNet_extractor(layers=num_layers).cuda()
    feature_model = feature_model.eval()

    val_dfs = []
    test_dfs = []

    # get attention weights in a df
    for i in range(3):
        model = Attn_Net_Gated(dim,hidden_dim,hidden_dim2,use_b,1)
        model.load_state_dict(torch.load(path+model_name+'-TCGA-'+str(i)+'-.pt'))
        model.eval()
        model = model.to('cuda')

        # validation
        df = get_performances_attn(feature_model, model, dataloaders['val_'+str(i)], image_transforms['valid'], \
                                    mapping_dict, db_path)
        df.to_csv(save_path+'/'+'val_df_'+str(i)+'.csv')
        val_dfs.append(df)

        # test
        df = get_performances_attn(feature_model, model, dataloaders['test'], image_transforms['valid'], \
                                 mapping_dict, db_path)
        df.to_csv(save_path+'/'+'test_df_'+str(i)+'.csv')
        test_dfs.append(df)