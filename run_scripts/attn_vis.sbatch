#!/usr/bin/bash
#SBATCH --job-name=attn_seed_512
#SBATCH --output=attn_seed_512.%j.out
#SBATCH --error=attn_seed_512.%j.err
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4

# load the module
ml load python/3.9.0

# activate virtualenv
source ~/code/wsi_mutation/bin/activate
 
# run code
python ~/code/WSI_mutation/code/AttentionVis.py --run-name attn_seed_512 \
                                                --all-regions False