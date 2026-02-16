#!/bin/bash 
#SBATCH --job-name=se
#SBATCH --time=72:00:00
#SBATCH --output=se-%a.out
#SBATCH --error=se-%a.out
#SBATCH --mem=32G
#SBATCH -p snsm_itn19
#SBATCH --array=0-29

## SBATCH --open-mode=append

module purge

config='semantic'

seed=$SLURM_ARRAY_TASK_ID
datasets=('r_1' 'r_2' 'keijzer_3' 'keijzer_4' 'keijzer_11' 'nguyen_12' 'pagie_1' 'vladislavleva_1' 'koza_3' 'keijzer_6' 'vladislavleva_8' 'korns_13' 'korns_14' 'korns_15')

for dataset in "${datasets[@]}"; do
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
done 