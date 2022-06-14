#!/usr/bin/bash
#SBATCH --job-name=allt2021
#SBATCH --output=allt2021.%j.out
#SBATCH --error=allt2021.%j.err
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=10
#SBATCH -C GPU_SKU:V100_PCIE

# load the module
ml load python/3.9.0
ml load py-scikit-learn/1.0.2_py39

# activate virtualenv
source ~/code/wsi_mutation/bin/activate
 
# run code
python3.9 ~/code/WSI_mutation/code/Eval_TCGA.py --dest-folder allt_seed_2021 \
                                                --model-folder allt_seed_2021 \
                                                --data-folder /oak/stanford/groups/ogevaert/data/Prad-TCGA/ \
                                                --annot False \
                                                --all-regions False \
                                                --use-db True \
                                                --source tcga \
                                                --precalc False \
                                                --gpu-num 0