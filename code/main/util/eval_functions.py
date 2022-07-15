import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def get_eval_df(patient_ids, labels, probs, preds, tile_paths):
    patient_ids_flatten = [j for sub in patient_ids for j in sub]
    labels_flatten = [j for sub in labels for j in sub]
    probs_flatten = [j for sub in probs for j in sub]
    preds_flatten = [j for sub in preds for j in sub]
    tile_paths_flatten = [j for sub in tile_paths for j in sub]
    eval_df = pd.DataFrame([patient_ids_flatten,labels_flatten,probs_flatten,preds_flatten, tile_paths_flatten]).transpose()
    eval_df.columns = ['patient_id', 'label', 'prob', 'pred', 'tile_path']
    
    return eval_df


def get_tile_performances(model, dataloader):
    patient_ids = []
    labels = []
    probs = []
    preds = []
    tile_paths = []

    final_auc = 0
    amount_not_in_auc = 0
    model.eval()

    with torch.no_grad():

        for ii, (data, target, patient_id, tile_path) in enumerate(dataloader):

            data = data.cuda()
            output = model(data)
            prob = torch.sigmoid(output)
            pred = torch.round(prob) 

            patient_ids.append(list(patient_id))
            labels.append(list(target.detach().numpy()))
            probs.append(list(prob.cpu().detach().numpy().flatten()))
            preds.append(list(pred.cpu().detach().numpy().flatten()))
            tile_paths.append(list(tile_path))

            if len(np.unique(target.detach().numpy().flatten())) > 1:
                auc = roc_auc_score(target.detach().numpy().flatten(),prob.cpu().detach().numpy().flatten())
                final_auc += auc * data.size(0)
            else: # in the final batch it can happen that there is only one label (e.g. 4 times 0 -> can't calculate auc)
                amount_not_in_auc = data.size(0)
                
        final_auc = final_auc / (len(dataloader.dataset)-amount_not_in_auc)
    
    return final_auc, patient_ids, labels, probs, preds, tile_paths   


def get_tile_performances_UZ(model, dataloader):
    patient_ids = []
    labels = []
    probs = []
    preds = []
    tile_paths = []
    matching_ids = []
    filenames = []

    test_auc = 0
    amount_not_in_auc = 0
    model.eval()

    for ii, (data, target, patient_id, tile_path, matching_id, filename) in enumerate(dataloader):
    
        data = data.cuda()
        output = model(data)
        prob = torch.sigmoid(output)
        pred = torch.round(prob) 

        patient_ids.append(list(patient_id))
        labels.append(list(target.detach().numpy()))
        probs.append(list(prob.cpu().detach().numpy().flatten()))
        preds.append(list(pred.cpu().detach().numpy().flatten()))
        tile_paths.append(list(tile_path))
        matching_ids.append(list(matching_id))
        filenames.append(list(filename))
        
        # for debugging
        if len(np.unique(target.detach().numpy().flatten())) > 1:
            auc = roc_auc_score(target.detach().numpy().flatten(),prob.cpu().detach().numpy().flatten())
            test_auc += auc * data.size(0)
        else: # in the final batch it can happen that there is only one label (e.g. 4 times 0 -> can't calculate auc)
            amount_not_in_auc = data.size(0)

    final_auc = test_auc / (len(dataloader.dataset)-amount_not_in_auc)
    
    return final_auc, patient_ids, labels, probs, preds, tile_paths, matching_ids, filenames


def get_eval_df_UZ(patient_ids, labels, probs, preds, tile_paths, matching_ids, filenames):
    patient_ids_flatten = [j for sub in patient_ids for j in sub]
    labels_flatten = [j for sub in labels for j in sub]
    probs_flatten = [j for sub in probs for j in sub]
    preds_flatten = [j for sub in preds for j in sub]
    tile_paths_flatten = [j for sub in tile_paths for j in sub]
    matching_ids_flatten = [j for sub in matching_ids for j in sub]
    filenames_flatten = [j for sub in filenames for j in sub]
    eval_df = pd.DataFrame([patient_ids_flatten,labels_flatten,probs_flatten,preds_flatten, tile_paths_flatten, matching_ids_flatten, filenames_flatten]).transpose()
    eval_df.columns = ['patient_id', 'label', 'prob', 'pred', 'tile_path', 'matching_id', 'filename']
    
    return eval_df

