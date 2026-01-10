#!/bin/bash 
#SBATCH --job-name=koza1
#SBATCH --time=72:00:00
#SBATCH --output=koza1-%a.out
#SBATCH --mem=32G
#SBATCH -p snsm_itn19
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-29%5

## SBATCH --open-mode=append

module purge

seed=$SLURM_ARRAY_TASK_ID
dataset='koza_1'
config='koza_set1'

echo "Running t-search with dataset=$dataset, config=$config, seed=$seed"

$HOME/t-search/t-search-env/bin/python -m t_search \
    --dataset $dataset \
    --config $HOME/t-search/configs/$config.json \
    --output $WORK_BGFS/t-search/$config-$dataset-results.jsonlist \
    --device cuda \
    --dtype float32 \
    --seed $seed

#srun --partition=snsm_itn19 --mem=4gb --time=01:00:00 --gpus=1 --pty bash -i