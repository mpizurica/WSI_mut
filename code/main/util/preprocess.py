from collections import defaultdict
from collections import Counter
import numpy as np
import random
import pandas as pd
import re

import lmdb
import pickle
import lz4framed
import csv
from PIL import Image

# set up logging
import logging
logger = logging.getLogger('tcga_logger.' + __name__)

# code written by Jakub Wasikowski
# url: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
def stratified_group_k_fold(X, y, groups, k, seed=None):
    """
    Creates grouped, stratified splits for data.
    :param X: input array
    :param y: labels for input array
    :param groups: array giving groups for input array
    :param k: number of folds
    :param seed: random seed
    """
    y=y.astype(int).tolist()
    labels_num = np.max(y) + 1

    # per person-> the amount of each sign is counted
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
  
    # the amount of the different signs is counted
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)
 
    # adds the counts of the signs of a random person to the current fold 
    # -> calculates the standard deviation for each sign in comparaison with 
    # the distribution of the signs in the total data set -> takes mean of it
    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
  
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)
  
    # in each iteration you add a person to one of the folds 
    # -> when added to a particular fold it gives the lowest difference with 
    # the original distribution!
    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])): # sorts groups_and_y_counts based on std from y_counts
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)
        
    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def read_ldmb(path, mapping_dict, img_name_wanted):
    lmdb_connection = lmdb.open(path,subdir=False, readonly=True, lock=False, readahead=False, meminit=False)

    ind = mapping_dict[img_name_wanted]

    with lmdb_connection.begin(write=False) as lmdb_txn:
        keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
        key = keys[ind]
        val = lmdb_txn.get(key)
        img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(val))
        if img_name.decode() != img_name_wanted:
            print('mapping dataframe not correct')
            return
        else:
            image = Image.fromarray(np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape))
            return image


def get_slidename(tile_path):
    s = tile_path.split('/')
    slidename = [i for i in s if ('.svs' in i and '.jpg' not in i)]
    return slidename[0]


def get_dataframe(data_folder, suffixes, gene, tiles_from_annot,\
                  all_regions, attention, train_or_test):

    gene_suffix = suffixes[gene]

    if attention:
        attention_suffix = suffixes['attention']
    else:
        attention_suffix = ''

    # setup df name
    if (tiles_from_annot == True) and (all_regions == False):
        regions_suffix = ''

    elif (tiles_from_annot == True) and (all_regions == True):
        regions_suffix = suffixes['all_regions']
        
    elif (tiles_from_annot == False):
        regions_suffix = suffixes['all_tiles']

    df_name = 'df_patients_labels'+gene_suffix+'_'+regions_suffix+attention_suffix+'_'+train_or_test+'.csv'

    # get train dataframe
    complete_df = pd.read_csv(data_folder+'TCGA_training/'+df_name, index_col=0) 
    logging.info('Used dataframe: %s\n' %df_name)
    print('Used dataframe: %s' %df_name)
    logging.info('Dataframe shape: %d\n' %(complete_df.shape[0]))

    if attention:
        complete_df['slide_name'] = complete_df.apply(lambda row: get_slidename(row['tile_path']),axis=1)

    # remap tile path for GPUlab
    path_to_tiles = data_folder+'TCGA_tiles/'
    if tiles_from_annot:
        complete_df['tile_path_remapped'] = complete_df['tile_path'].apply(lambda x: re.sub('/data/thesis_2021/PRAD_dx/',path_to_tiles,x))
    else:
        complete_df['tile_path_remapped'] = complete_df['tile_path'].apply(lambda x: re.sub('/home/bioit/mpizuric/Documents/',path_to_tiles,x))  

    # rename columns for tile paths
    complete_df.rename(columns = {'tile_path':'tile_path_old'}, inplace = True)
    complete_df.rename(columns = {'tile_path_remapped':'tile_path'}, inplace = True) 

    return complete_df


def undersample_df_nr_tiles_pp_random(df, amount=500, random_state=0):
    """
    Undersample the number of tiles per person to a certain amount - entirely randomly.
    :param df: dataframe to undersample. Must contain column patient_id
    :param amount: amount of tiles to keep per patient
    """
    grouped = df.groupby(df['patient_id'])
    dataframes = []

    # grouped dataframes by patient
    for k,g in grouped:
        
        if g.shape[0] > amount:

            amount_to_remove = int(g.shape[0]-amount)
            samples = g.sample(n=amount_to_remove, random_state=random_state)
            # remove samples from dataframe
            g = pd.concat([g, samples]).drop_duplicates(keep=False)
            
        # add dataframe of this patient to list of undersampled dfs
        dataframes.append(g)
            
    return pd.concat(dataframes)    
            

def calc_undersample_factor_random(df, gene, column='patient_id'):
    """
    Calculates the amount of tiles per patient we can have in the no mutation class. See formula 6.3 in thesis.

    :param df: dataframe which contains patient_id (or other column for patient, see column variable) and class label in "gene" column
    :param gene: column in df which contains class label (0 or 1), with 0 being the MAJORITY label
    :param column: column in df which contains patient identifier
    """

    # count number of patients without mut
    maj_df = df.loc[df[gene]==0]
    amt_patients = len(np.unique(maj_df['patient_id'].values))
    
    return np.round(len(df.loc[df[gene]==1].values)/amt_patients)
    
    
def undersample_df_random(df, gene, maj_label=0, min_label=1, amount=100, random_state=0):
    """
    Undersample df for class balance. 

    :param df: df which needs to be undersampled for class balance
    :param gene: column in df of gene for which there is unbalanced data
    :param maj_label: gene label for which most data is available
    :param min_label: gene label for which least data is available
    :param amount: minimum amount of tiles to keep for a patient (if this is equal to the value returned by calc_undersample_factor, you'll get class balance)
    :param random_state: random init state
    """
    amt_maj_class = df.loc[df[gene]==maj_label].shape[0]
    amt_min_class = df.loc[df[gene]==min_label].shape[0]
    
    grouped = df.groupby(df['patient_id'])

    dataframes = []

    # grouped dataframes by patient
    for k,g in grouped:

        condition = (g.shape[0] > amount) and (g.iloc[0][gene]!=1) and (amt_maj_class > amt_min_class)

        # if amount of tiles per patient is larger than amount allowed
        # only do this if the patient does not have a mutation
        if condition:

            # amount to remove for this patient: amount of tiles it has on top of minimal amount
            amount_to_remove = int(g.shape[0]-amount)

            # only remove the amount necessary for class balance
            if (amt_maj_class - amt_min_class) < amount_to_remove:
                amount_to_remove = amt_maj_class - amt_min_class

            if g.shape[0] > amount_to_remove:
                samples = g.sample(n=amount_to_remove, random_state=random_state)
                # remove samples from dataframe
                g = pd.concat([g, samples]).drop_duplicates(keep=False)
                amt_maj_class -= amount_to_remove
            
        # add dataframe of this patient to list of undersampled dfs
        dataframes.append(g)
            
    return pd.concat(dataframes)


def undersample_df_nr_tiles_pp(df, amount=500, random_state=0):
    """
    Undersample the number of tiles per person to a certain amount. 
    CAUTION by default, this method will keep "amount" tiles per person PER GLEASON SCORE (did not make a difference until now because we only used the highest graded regions per patient)

    :param df: dataframe to undersample. Must contain column patient_id, tile_path
    :param amount: amount of tiles to keep per patient
    """

    grouped = df.groupby(df['patient_id'])
    dataframes = []

    # grouped dataframes by patient
    for k,g in grouped:
        
        if g.shape[0] > amount:

            # take into account different gleason score tiles
            gleason_scores_patient = {}
            gleason_scores_tiles = {}
            
            for i in g['tile_path'].values:
                
                if 'GS' in i:
                    GS = ((i.split('GS')[1]).split('/')[0])
                else:
                    GS = i.split('_')[-2]
                    
                if GS not in gleason_scores_patient:
                    gleason_scores_patient[GS]=1
                    gleason_scores_tiles[GS] = [i]
                else:
                    gleason_scores_patient[GS]+=1
                    gleason_scores_tiles[GS].append(i)

            # randomly remove tiles that have > amount tiles for a certain gleason score
            for GS in gleason_scores_patient:
                if gleason_scores_patient[GS] > amount:
                    gs_df = g.loc[g['tile_path'].str.contains(GS,regex=False)]
                    amount_to_remove = int(gleason_scores_patient[GS]-amount)

                    samples = gs_df.sample(n=amount_to_remove, random_state=random_state)
                    # remove samples from dataframe
                    g = pd.concat([g, samples]).drop_duplicates(keep=False)
            
        # add dataframe of this patient to list of undersampled dfs
        dataframes.append(g)
            
    return pd.concat(dataframes)    
    

def calc_undersample_factor(df, gene, column='patient_id'):
    """
    Calculates the amount of tiles per patient we can have in the no mutation class. See formula 6.3 in thesis.

    :param df: dataframe which contains tile_path, patient_id (or other column for patient, see column variable) and class label in "gene" column
    :param gene: column in df which contains class label (0 or 1), with 0 being the MAJORITY label
    :param column: column in df which contains patient identifier
    """

    # count number of gleason scores per patient 
    # we will calculate how many tiles every gs can contain in order to have balanced data
    maj_df = df.loc[df[gene]==0]
    grouped = maj_df.groupby(maj_df[column])
    amt_gs = 0

    # this for loop is only necessary if we keep multiple gleason scores per patient
    # if only highest graded region --> this loop will return the amount of patients 
    # (else we count the distinct gleason scores for all patients)
    # grouped dataframes by patient
    for k,g in grouped:
        gleason_scores_patient = []
        for i in g['tile_path'].values:
            if 'GS' in i:
                GS = ((i.split('GS')[1]).split('/')[0])
            else:
                GS = i.split('_')[-2]
            #GS = ((i.split('GS')[1]).split('/')[0]) 
            if GS not in gleason_scores_patient:
                gleason_scores_patient.append(GS)
        amt_gs += len(gleason_scores_patient)
    
    return np.round(len(df.loc[df[gene]==1].values)/amt_gs)
    
    
def undersample_df(df, gene, maj_label=0, min_label=1, amount=100, random_state=0):
    """
    Undersample df for class balance. The code takes into account the case where you want to keep multiple gleason scores per 
    patient, but we just used this for the highest graded regions for patients

    :param df: df which needs to be undersampled for class balance
    :param gene: column in df of gene for which there is unbalanced data
    :param maj_label: gene label for which most data is available
    :param min_label: gene label for which least data is available
    :param amount: minimum amount of tiles to keep for a patient (if this is equal to the value returned by calc_undersample_factor, you'll get class balance)
    :param random_state: random init state
    """
    amt_maj_class = df.loc[df[gene]==maj_label].shape[0]
    amt_min_class = df.loc[df[gene]==min_label].shape[0]
    
    grouped = df.groupby(df['patient_id'])

    dataframes = []

    # grouped dataframes by patient
    for k,g in grouped:

        condition = (g.shape[0] > amount) and (g.iloc[0][gene]!=1) and (amt_maj_class > amt_min_class)

        # if amount of tiles per patient is larger than 100
        # only do this if the patient does not have a mutation
        if condition:

            # take into account different gleason score tiles
            gleason_scores_patient = {}
            gleason_scores_tiles = {}
            for i in g['tile_path'].values:
                
                if 'GS' in i:
                    GS = ((i.split('GS')[1]).split('/')[0])
                else:
                    GS = i.split('_')[-2]
                    
                if GS not in gleason_scores_patient:
                    gleason_scores_patient[GS]=1
                    gleason_scores_tiles[GS] = [i]
                else:
                    gleason_scores_patient[GS]+=1
                    gleason_scores_tiles[GS].append(i)

            # randomly remove tiles that have > amount (default 100) tiles for a certain gleason score
            for GS in gleason_scores_patient:
                if gleason_scores_patient[GS] > amount:
                    gs_df = g.loc[g['tile_path'].str.contains(GS,regex=False)]
                    gs_df = gs_df.loc[gs_df[gene] == maj_label]
                    amount_to_remove = int(gleason_scores_patient[GS]-amount)
                    if (amt_maj_class - amt_min_class) < amount_to_remove:
                        amount_to_remove = amt_maj_class - amt_min_class
                    if gs_df.shape[0] > amount_to_remove:
                        samples = gs_df.sample(n=amount_to_remove, random_state=random_state)
                        # remove samples from dataframe
                        g = pd.concat([g, samples]).drop_duplicates(keep=False)
                        amt_maj_class -= amount_to_remove
            
        # add dataframe of this patient to list of undersampled dfs
        dataframes.append(g)
            
    return pd.concat(dataframes)


def get_undersampled_df(df, amt_tiles_pp, gene, tiles_from_annot):

    if tiles_from_annot:
        # undersample to 500 tiles per patient, take into account gleason grade
        temp_df = undersample_df_nr_tiles_pp(df, amount=amt_tiles_pp, \
                                                random_state=0).sample(frac = 1) 
        # undersample for class balance, take into account gleason grade
        temp_df_2 = undersample_df(temp_df, \
                                    amount=calc_undersample_factor(temp_df, gene), gene=gene,\
                                    random_state=0).sample(frac = 1)  

        temp_df_2 = temp_df_2.reset_index(drop=True)

    else:
        # undersample to 500 tiles per patient, entirely randomly
        temp_df = undersample_df_nr_tiles_pp_random(df, amount=amt_tiles_pp, \
                                                random_state=0).sample(frac = 1) 
        # undersample for class balance, entirely randomly
        temp_df_2 = undersample_df_random(temp_df, \
                                    amount=calc_undersample_factor(temp_df, gene), gene=gene,\
                                    random_state=0).sample(frac = 1)  

        temp_df_2 = temp_df_2.reset_index(drop=True)


    return temp_df_2


def get_undersampled_dfs(dfs, tiles_from_annot, amt_tiles_pp, gene):
    """
    undersampled dataframes, multiple for different epochs
    # IMPORTANT: do NOT reset index!!! indices will be used to trace back samples to use from original df
    # each list contains indices for different train sets which are diff. undersamplings
    """
    df_ind_epoch = [[],[],[]]

    for i in range(3):
        for j in range(30):
            if tiles_from_annot:
                # undersample to 500 tiles pp, different random state each epoch
                # make sure to shuffle indices again (this function returns dataframe with patients in order)
                temp_df = undersample_df_nr_tiles_pp(dfs[i], amount=amt_tiles_pp, \
                                                        random_state=j).sample(frac = 1) 
                
                # different random state each epoch
                # make sure to shuffle indices again (this function returns dataframe with patients in order)
                temp_df_2 = undersample_df(temp_df, \
                                            amount=calc_undersample_factor(temp_df, gene), gene=gene,\
                                            random_state=j).sample(frac = 1)   

                df_ind_epoch[i].append(temp_df_2.index.values)  
            else:
                    # undersample to 500 tiles pp, different random state each epoch
                # make sure to shuffle indices again (this function returns dataframe with patients in order)
                temp_df = undersample_df_nr_tiles_pp_random(dfs[i], amount=amt_tiles_pp, \
                                                        random_state=j).sample(frac = 1) 
                
                # different random state each epoch
                # make sure to shuffle indices again (this function returns dataframe with patients in order)
                temp_df_2 = undersample_df_random(temp_df, \
                                                amount=calc_undersample_factor(temp_df, gene), gene=gene,\
                                                random_state=j).sample(frac = 1)   

                df_ind_epoch[i].append(temp_df_2.index.values)   

    return df_ind_epoch

    

