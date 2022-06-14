#!/usr/bin/bash
#SBATCH --job-name=attn_allt512
#SBATCH --output=attn_allt512.%j.out
#SBATCH --error=attn_allt512.%j.err
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=20
#SBATCH -C GPU_SKU:V100_PCIE

# load the module #GPU_SKU:V100_SXM2
ml load python/3.9.0
ml load py-scikit-learn/1.0.2_py39

# activate virtualenv
source ~/code/wsi_mutation/bin/activate
 
# run code
python3.9 ~/code/WSI_mutation/code/TCGA_training.py --dest-folder None \
                                                    --annot False \
                                                    --attention True \
                                                    --all-regions False \
                                                    --seed-crossval 512  \
                                                    --batch-size 32 \
                                                    --lr 2e-4 \
                                                    --num-epochs 30 \
                                                    --use-scheduler None \
                                                    --data-folder /oak/stanford/groups/ogevaert/data/Prad-TCGA/ \
                                                    --use-db True \
                                                    --gpu-num 0 \
                                                    --num-workers 20 \
                                                    --multiple-undersamplings False \
                                                    --undersample-validation True \
                                                    --undersample-train True \
                                                    --train-folds 0,1,2 \
                                                    --indices-path None 