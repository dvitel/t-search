#!/bin/bash 
#SBATCH --job-name=pof
#SBATCH --time=72:00:00
#SBATCH --output=pof-%a.out
#SBATCH --error=pof-%a.out
#SBATCH --mem=32G
#SBATCH --gpus=1 # 1 GPU
#SBATCH -p snsm_itn19
#SBATCH --array=0-2
#SBATCH --exclude=mdc-1057-28-15,mdc-1057-27-18

## SBATCH --open-mode=append

module purge

config='point_optim_2rr'

seeds=(16 21 24)
datasets=('korns_15' 'korns_15' 'korns_13')
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "====================================================================="
echo "Running t-search with dataset=$dataset, config=$config, seed=$seed"

$HOME/t-search/t-search-env/bin/python -m t_search \
    --dataset $dataset \
    --config $HOME/t-search/configs/$config.json \
    --output $WORK_BGFS/t-search/results.jsonlist \
    --device cuda \
    --dtype float32 \
    --seed $seed

echo "Finished t-search with dataset=$dataset, config=$config, seed=$seed"