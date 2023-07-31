# Whole slide imaging-based prediction of TP53 mutations identifies an aggressive disease phenotype in prostate cancer 

[Journal link](https://aacrjournals.org/cancerres/article/doi/10.1158/0008-5472.CAN-22-3113/727461/Whole-slide-imaging-based-prediction-of-TP53) | Cite

In prostate cancer, there is an urgent need for objective prognostic biomarkers that identify the metastatic potential of a tumor at an early stage. While recent analyses indicated TP53 mutations as candidate biomarkers, molecular profiling in a clinical setting is complicated by tumor heterogeneity. Deep learning models that predict the spatial presence of TP53 mutations in whole slide images (WSIs) offer the potential to mitigate this issue. To assess the potential of WSIs as proxies for spatially resolved profiling and as biomarkers for aggressive disease, we developed TiDo, a deep learning model that achieves state-of-the-art performance in predicting TP53 mutations from WSIs of primary prostate tumors. In an independent multi-focal cohort, the model showed successful generalization at both the patient and lesion level. Analysis of model predictions revealed that false positive (FP) predictions could at least partially be explained by TP53 deletions, suggesting that some FP carry an alteration that leads to the same histological phenotype as TP53 mutations. Comparative expression and histological cell type analyses identified a TP53-like cellular phenotype triggered by expression of pathways affecting stromal composition. Together, these findings indicate that WSI-based models might not be able to perfectly predict the spatial presence of individual TP53 mutations but they have the potential to elucidate the prognosis of a tumor by depicting a downstream phenotype associated with aggressive disease biomarkers.

# WSI annotations and processing

![](images/annotations.png?raw=true)

Detailed, expert-based WSI annotations for TCGA-PRAD are available in the [tcga](https://github.com/mpizurica/WSI_mut/tree/master/code/data_prep/tcga) folder.
The folder also includes code for 
- WSI tiling
- filtering blurred tiles and tiles with pen marks
- creating a dataframe that contains all tiles with corresponding _TP53_ mutation labels
- splitting off a held-out test set

## Slides and tiles
- slides can be downloaded on the GDC data portal (https://portal.gdc.cancer.gov/repository). 
- slide filenames discarded due to bad quality can be found in [bad_quality_slides.txt](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/bad_qual_slides.txt)
- we generated tiles from annotations made in QuPath (https://qupath.github.io/). See qupath_tiling.groovy for code. The tiles can also be generated in python with the annotation masks we provide (see below). If tiling in python, the resulting file structure (that corresponds to structure expected by code for later dataframe creation) should be
```
.
|-- slide_name1.svs
         |-- GSx+y 
               |-- slide_name1.svs_coordx;coordy_GSx+y_.jpg 
```
with coordx and coordy the coordinates for the tile and GSx+y the annotated Gleason Grade of the region where the tile was extracted from.
- annotation masks are provided in [masks_TCGA.zip](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/masks_TCGA.zip). They were downsampled with __factor 64__ and they are color coded to show the annotated Gleason Grade. The mapping for colors to Gleason Grade is as follows (where 'Tumor' denotes a region without annotated grade)
```
mapping = {'(157, 13, 45)':'GS4+4', '(156, 1, 115)':'GS4+5', \
          '(119, 95, 233)':'Tumor', '(64, 197, 186)':'GS3+5', \
          '(36, 95, 137)':'GS5+5', '(157, 200, 2)':'GS4+3', \
          '(64, 79, 16)':'GS3+3','(64, 128, 142)':'GS3+4', \
          '(37, 38, 23)':'GS5+3','(36, 26, 93)':'GS5+4'}
```

## Mutation data
- see [TCGA mutect](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/TCGA.PRAD.mutect.deca36be-bf05-441a-b2e4-394228f23fbe.DR-10.0.somatic.tar.xz) tar file and [Strelka2.tar.xz](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/Strelka2.tar.xz) file for mutation labels (are used in [prepare_dataframe.py](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/prepare_dataframe.py)) (the Strelka2.tar.xz contains PRAD_pyclone_snvs.tsv and PRAD_indels_annotated.tsv which are used in the prepare_dataframe.py code)

## Preparing dataframe and tile quality filtering
- after tiling, you should have a folder that contains tiles for all slides, with folder structure as described above
- now use [prepare_dataframe.py](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/prepare_dataframe.py) to create the TCGA dataframe. It will contain all tiles for patients with corresponding TP53 mutation label (it collects tiles from 'source' folder).
- [filter_tiles.py](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/filter_tiles.py) can be used to identify white, blurred, penmarked tiles in the dataframe if you want to remove these. This script will generate a text file which contains tile paths that are undesirable (which can be later used in the training script)
- now, [train_test_split.py](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/train_test_split.py) can be used to split the data in train (train+validation) and held-out test set


# Model

![](images/model.png?raw=true)

Code for training the model is available in the [main](https://github.com/mpizurica/WSI_mut/tree/master/code/main) folder. Code for running the scripts are provided in the 
[run_scripts](https://github.com/mpizurica/WSI_mut/tree/master/run_scripts) folder.

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
  --use-db              If `True`, the code assumes you will use .db files that contain all tiles of a specific slide (instead of saving tiles separately in .jpg or .png). See lmdb_creation folder for more info. If `True`, then the code expects the databases to be stored in  data_folder/TCGA_tiles/db_tiles_512px/ and the code also expects a dataframe `img_name_to_index.csv` which contains the index of every tile within the LMDB of a given slide (for easier and faster tile processing in the dataset class - see code)
  ```
  
- `eval_TCGA.py`: evaluating tile-level model on TCGA. Will use the model stored in data_folder/TCGA_training/runs/TCGA_train/model_run_folder. The code returns dataframes containing predictions for all tiles in validation and test set.
- `eval_UZ.py`: evaluates tile-level model on held-out test set from UZ Ghent.
- `eval_TCGA_attn.py`: evaluating attention-based model on TCGA. The code returns dataframes containing predictions for all slides in validation and test set, and includes the attention weight per tile.

Checkpoints for _BeTiDo_ (see paper) can be downloaded on [this link](https://drive.google.com/drive/folders/18-zx6tSvv6M3INFP_UvgLYSmeXqzaiUj?usp=sharing).

# Interpretations

![](images/interpretations.png?raw=true)

To generate a tile as in (a): (see [notebook](https://github.com/mpizurica/WSI_mut/tree/master/code/gradcam))
1. Use [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) to obtain (binary) grad-cam mask
2. For cell type detection in tiles see [HoverNet](https://github.com/vqdang/hover_net)
3. The contour of the binary gradcam mask (from 1.) can be visualized on top of the tile (from 2.) as follows:
 
```
def show_cam_on_image_contour(img, mask, thresh=50):
    mask_ = (mask*255).astype(np.uint8)
    ret,thresh_img = cv2.threshold(mask_, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cont_im = img
    cv2.drawContours(cont_im, contours, -1, (0,0,0), 3)
    
    return cont_im
```

Code for differential gene expression analysis can be found in the [diff_expr](https://github.com/mpizurica/WSI_mut/tree/master/code/diff_expr) folder. To exectute:

1. prepare dataframe that contains two columns with the two groups you wish to compare (the column should contain the relevant patient ids)
2. use [prepare_de_files.py](https://github.com/mpizurica/WSI_mut/blob/master/code/diff_expr/prepare_de_files.py) to prepare tsv files which will be used by the script that performs the differential expression analysis
3. use resulting tsv files as input in [DE_analysis.Rmd](https://github.com/mpizurica/WSI_mut/blob/master/code/diff_expr/DE_analysis.Rmd)
4. [analyze_DE_result.py](https://github.com/mpizurica/WSI_mut/blob/master/code/diff_expr/analyze_DE_result.py) allows to analyse the resulting genes 

- necessary files used in the scripts can be found in [needed_files.zip](https://github.com/mpizurica/WSI_mut/blob/master/code/diff_expr/needed_files.zip)
- htseq counts used in [prepare_de_files.py](https://github.com/mpizurica/WSI_mut/blob/master/code/diff_expr/prepare_de_files.py) can be downloaded from Xenahubs ([link](https://xenabrowser.net/datapages/?dataset=TCGA-PRAD.htseq_counts.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443))

# Citation
