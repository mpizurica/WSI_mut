# Differential expression analysis

1. prepare dataframe that contains two columns with the two groups you wish to compare (the column should contain the relevant patient ids)
2. use [prepare_de_files.py](https://github.com/mpizurica/WSI_mut/blob/master/code/diff_expr/prepare_de_files.py) to prepare tsv files which will be used by the script that performs the differential expression analysis
3. use resulting tsv files as input in [DE_analysis.Rmd](https://github.com/mpizurica/WSI_mut/blob/master/code/diff_expr/DE_analysis.Rmd)
4. [analyze_DE_result.py](https://github.com/mpizurica/WSI_mut/blob/master/code/diff_expr/analyze_DE_result.py) allows to analyse the resulting genes 
