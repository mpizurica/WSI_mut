## Slides and tiles
- slides can be downloaded on the GDC data portal (https://portal.gdc.cancer.gov/repository). 
- slide filenames discarded due to bad quality can be found in bad_quality_slides.txt
- we generated tiles from annotations made in QuPath (https://qupath.github.io/). See qupath_tiling.groovy for code. The tiles can also be generated in python with the annotation masks we provide (see below). Make sure to save the gleason grade in the tile name if you want to select tiles from specific lesions later in training (or e.g. only export tiles from lesions with specific grade)
- annotation masks are provided in masks_TCGA.zip. They were downsampled with __factor 64__ (important if tiling via python) and they are color coded to show the annotated Gleason Grade. The mapping for colors to Gleason Grade is as follows (where 'Tumor' denotes a region without annotated grade)
```
mapping = {'(157, 13, 45)':'GS4+4', '(156, 1, 115)':'GS4+5', \
          '(119, 95, 233)':'Tumor', '(64, 197, 186)':'GS3+5', \
          '(36, 95, 137)':'GS5+5', '(157, 200, 2)':'GS4+3', \
          '(64, 79, 16)':'GS3+3','(64, 128, 142)':'GS3+4', \
          '(37, 38, 23)':'GS5+3','(36, 26, 93)':'GS5+4'}
```

## Mutation data
- see TCGA mutect tar file and excel with labels from Kather et al. (2020) (https://doi.org/10.1038/s43018-020-0087-6) 
- resulting labels per patient are given in label_df.csv (patients with inconsistent mutation labels discarded, as defined in paper)

