# Code for training and evaluating model

How data should be structured to run the scripts

```
    |--data_folder
           |--TCGA_tiles
                  |-- slide_name1.svs
                        |-- GSx+y 
                              |-- slide_name1.svs_coordx;coordy_GSx+y_.jpg 
                               ...
                   |-- slide_name2.svs
                   
           |--TCGA_training
                  |--df_patients_labels.csv
                  |--blurred_tile_paths.txt
                  ...
                  |--runs
                       |--TCGA-train
                       |--TCGA-eval
```

files:

- `TCGA_training.py`: training tile-level or attention-based model. You can set the desired level of annotation detail (dominant regions, all tumor regions, whole slide)
  - Example on how to run provided in [run_scripts](https://github.com/mpizurica/WSI_mut/tree/master/run_scripts)
  - Note that in the current implementation, setting a certain level of annotation detail is achieved by selecting the dataframe with the correct filename (e.g. dataframe
which only contains tiles from dominant tumor region, as created in 
[prepare_dataframe.py](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/prepare_dataframe.py)). The specific format of the filename should be
`df_patients_labels<annotation_detail_suffix><attention_suffix><train_or_test>.csv` with:

        - <annotation_detail_suffix> = _all_tiles for the entire WSI, _all_regions for all annotated tumor regions, and empty for dominant tumor regions
        - <attention_suffix> = _attention if using the attention based model, otherwise empty
        - <train_or_test> = _TRAIN for the train dataframe (later split in train and validation), _TEST for the held-out test set (not used in this script)

  - Important arguments:
  
  ```
  --annot               Whether to use tiles from annotated regions. If `False`, then tiles from the whole slide image are used.
  --all-regions         only relevant if `annot` is `True`. If `all-regions` is `False`, then only tiles from the dominant tumor region will be included. Otherwise, all annotated regions are taken into account. 
  --attention           If `True`, attention-based model is used, otherwise tile-level model.
  --dest-folder         Results (model checkpoints, logfile...) will be saved in the dir provided in data_folder + '/TCGA_training/runs/TCGA_train/' + dest_folder
  ```
  

- `eval_TCGA.py`: evaluating tile-level model on TCGA. Will use the model stored in data_folder/TCGA_training/runs/TCGA_train/model_run_folder. The code returns dataframes containing predictions for all tiles in validation and test set.
- `eval_UZ.py`: evaluates tile-level model on held-out test set from UZ Ghent. 
- `eval_TCGA_attn.py`: evaluating attention-based model on TCGA. The code returns dataframes containing predictions for all slides in validation and test set, and includes the attention weight per tile.

Checkpoints for _BeTiDo_ (see paper) can be downloaded on [this link](https://drive.google.com/drive/folders/18-zx6tSvv6M3INFP_UvgLYSmeXqzaiUj?usp=sharing).

