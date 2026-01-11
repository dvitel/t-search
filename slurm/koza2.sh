#!/bin/bash 
#SBATCH --job-name=K2
#SBATCH --time=72:00:00
#SBATCH --output=K2-%a.out
#SBATCH --error=K2-%a.out
#SBATCH --mem=32G
#SBATCH -p snsm_itn19
#SBATCH --gpus=1 # 1 GPU
#SBATCH --array=0-29

## SBATCH --open-mode=append

module purge

config='koza_set2'

seed=$SLURM_ARRAY_TASK_ID
datasets=('koza_1' 'koza_2' 'koza_3' 'nguyen_1' 'nguyen_2' 'nguyen_3' 'nguyen_4' 'nguyen_5' 'nguyen_6' 'nguyen_7' 'nguyen_8' 'nguyen_9' 'nguyen_10' 'pagie_1' 'pagie_2' 'korns_1' 'korns_2' 'korns_3' 'korns_4' 'korns_5' 'korns_6' 'korns_7' 'korns_8' 'korns_9' 'korns_10' 'korns_11' 'korns_12' 'korns_13' 'korns_14' 'korns_15' 'keijzer_1' 'keijzer_2' 'keijzer_3' 'keijzer_4' 'keijzer_5' 'keijzer_6' 'keijzer_7' 'keijzer_8' 'keijzer_9' 'keijzer_10' 'keijzer_11' 'keijzer_12' 'keijzer_13' 'keijzer_14' 'keijzer_15' 'vladislavleva_1' 'vladislavleva_2' 'vladislavleva_3' 'vladislavleva_4' 'vladislavleva_5' 'vladislavleva_6' 'vladislavleva_7' 'vladislavleva_8')

for dataset in "${datasets[@]}"; do
    echo "Running t-search with dataset=$dataset, config=$config, seed=$seed"

    $HOME/t-search/t-search-env/bin/python -m t_search \
        --dataset $dataset \
        --config $HOME/t-search/configs/$config.json \
        --output $WORK_BGFS/t-search/results.jsonlist \
        --device cuda \
        --dtype float32 \
        --seed $seed
done 