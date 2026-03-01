#!/bin/bash 
#SBATCH --job-name=cof3
#SBATCH --time=72:00:00
#SBATCH --output=cof3-%a.out
#SBATCH --error=cof3-%a.out
#SBATCH --mem=64G
#SBATCH --gpus=1 # 1 GPU
#SBATCH -p snsm_itn19
#SBATCH --array=0-2
#SBATCH --exclude=mdc-1057-28-15,mdc-1057-27-18

## SBATCH --open-mode=append

module purge

config='competent'

seeds=(5 5 5)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
datasets=('korns_13' 'korns_14' 'korns_15')
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