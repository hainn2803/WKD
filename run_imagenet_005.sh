#!/bin/bash -e
#SBATCH --job-name=img_005inter
#SBATCH --output=/lustre/scratch/client/movian/research/users/hainn14/WKD/spp_noti/imagenet_005inter.out
#SBATCH --error=/lustre/scratch/client/movian/research/users/hainn14/WKD/spp_noti/imagenet_005inter.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=movianr
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io

module purge
module load python/miniconda3/miniconda3

# Corrected line
eval "$(conda shell.bash hook)"

conda activate /lustre/scratch/client/movian/research/users/hainn14/envs/wkd
cd /lustre/scratch/client/movian/research/users/hainn14/WKD

export TORCH_HOME=/lustre/scratch/client/movian/research/users/hainn14/WKD/download_ckpts

WANDB_MODE=disabled python tools/train.py --cfg configs/imagenet/r34_r18/wkd_f005.yaml --dataset /lustre/scratch/client/movian/research/users/hainn14/dataset/imagenet --MD 0.05
WANDB_MODE=disabled python tools/train.py --cfg configs/imagenet/r34_r18/wkd_f005.yaml --dataset /lustre/scratch/client/movian/research/users/hainn14/dataset/imagenet --MD 0.05