import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter
import random

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


def get_slidename(tile_path):
    s = tile_path.split('/')
    slidename = [i for i in s if ('.svs' in i and '.jpg' not in i)]
    return slidename[0]


if __name__ == '__main__':
    df_name = 'df_patients_labels.csv'
    df_patients_labels = pd.read_csv(df_name)
    gene = 'TP53'
    attention = False # if we use the attention based model, we will balance number of slides in splits, otherwise number of tiles (while keeping tiles/slides of same patient in same split)
    num_folds = 5 # we will use one of five folds (20%) as test set

    if attention:
        # if slide name is not yet in the columns
        df_patients_labels['slide_name'] = df_patients_labels.apply(lambda row: get_slidename(row['tile_path']),axis=1)

        df_patients_labels_temp = df_patients_labels.groupby(df_patients_labels['slide_name'])
        df_patients_labels_temp = df_patients_labels_temp.head(1)[['slide_name',gene,'patient_id']]

        # shuffle dataframe
        final_df = df_patients_labels_temp.sample(frac=1, random_state=0).reset_index(drop=True)

        # train, test, split
        for fold_ind, (dev_ind, val_ind) in enumerate(stratified_group_k_fold(final_df.index, \
                                                                                final_df[gene], \
                                                                                final_df["patient_id"], \
                                                                                num_folds, seed=2021)):
            if fold_ind == 0:
                complete_train_df = final_df.iloc[dev_ind] 
                test_df = final_df.iloc[val_ind] 

        test_df_ = df_patients_labels[df_patients_labels['patient_id'].isin(test_df['patient_id'].values)]
        complete_train_df_ = df_patients_labels[df_patients_labels['patient_id'].isin(complete_train_df['patient_id'].values)]
        test_df_.to_csv(df_name+'_attention_TEST.csv')
        complete_train_df_.to_csv(df_name+'_attention_TRAIN.csv')

    else:
        # shuffle dataframe
        final_df = df_patients_labels.sample(frac=1, random_state=0).reset_index(drop=True)

        # train, test, split
        for fold_ind, (dev_ind, val_ind) in enumerate(stratified_group_k_fold(final_df["tile_path"], \
                                                                                final_df[gene], \
                                                                                final_df["patient_id"], \
                                                                                num_folds, seed=2021)):
            if fold_ind == 0:
                complete_train_df = final_df.iloc[dev_ind] 
                test_df = final_df.iloc[val_ind] 

        test_df.to_csv(df_name+'_TEST.csv')
        complete_train_df.to_csv(df_name+'_TRAIN.csv')








