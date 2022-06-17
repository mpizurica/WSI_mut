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
- see [TCGA mutect](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/TCGA.PRAD.mutect.deca36be-bf05-441a-b2e4-394228f23fbe.DR-10.0.somatic.tar.xz) tar file and [Strelka2.tar.xz](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/Strelka2.tar.xz) file for mutation labels (are used in [prepare_dataframe.py](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/prepare_dataframe.py))

## Preparing dataframe and tile quality filtering
- after tiling, you should have a folder (referred to as 'source' in code) that contains tiles for all slides, with folder structure as described above
- now use [prepare_dataframe.py](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/prepare_dataframe.py) to create the TCGA dataframe. It will contain all tiles for patients with corresponding TP53 mutation label (it collects tiles from 'source' folder).
- [filter_tiles.py](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/filter_tiles.py) can be used to identify white, blurred, penmarked tiles in the dataframe if you want to remove these. This script will generate a text file which contains tile paths that are undesirable (which can be later used in the training script)
- now, [train_test_split.py](https://github.com/mpizurica/WSI_mut/blob/master/code/data_prep/tcga/train_test_split.py) can be used to split the data in train (train+validation) and held-out test set
