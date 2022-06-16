# Characterizing aggressive prostate cancer by using deep learning on Whole Slide Images

Characterizing aggressive prostate cancer by using deep learning on Whole Slide Images

Arxiv | Journal link | Cite

Nam dui ligula, fringilla a, euismod sodales, sollicitudin vel, wisi. Morbi auctor lorem non
justo. Nam lacus libero, pretium at, lobortis vitae, ultricies et, tellus. Donec aliquet, tortor sed
accumsan bibendum, erat ligula aliquet magna, vitae ornare odio metus a mi. Morbi ac orci et nisl
hendrerit mollis. Suspendisse ut massa. Cras nec ante. Pellentesque a nulla. Cum sociis natoque
penatibus et magnis dis parturient montes, nascetur ridiculus mus. Aliquam tincidunt urna. Nulla
ullamcorper vestibulum turpis. Pellentesque cursus luctus mauris

# WSI annotations and processing

![](images/annotations.png?raw=true)

Detailed, expert-based WSI annotations for TCGA-PRAD are available in the [tcga](https://github.com/mpizurica/WSI_mut/tree/master/code/data_prep/tcga) folder.
The folder includes code for 
- WSI tiling
- filtering blurred tiles and tiles with pen marks
- creating a dataframe that contains all tiles with corresponding _TP53_ mutation labels
- splitting off a held-out test set

# Model
<p align="center">
  <img src="https://github.com/mpizurica/WSI_mut/blob/master/images/model.png" alt="drawing" width="750"/>
 </p>
