#!/bin/bash 
#SBATCH --job-name=pof
#SBATCH --time=72:00:00
#SBATCH --output=pof-%a.out
#SBATCH --error=pof-%a.out
#SBATCH --mem=32G
#SBATCH --gpus=1 # 1 GPU
#SBATCH -p snsm_itn19
#SBATCH --array=0-7
#SBATCH --exclude=mdc-1057-28-15,mdc-1057-27-18

## SBATCH --open-mode=append

module purge

config='point_optim_nwf'

seeds=(16 23 24 25 26 27 28 29)
datasets=('r_1' 'r_2' 'keijzer_3' 'keijzer_4' 'keijzer_11' 'nguyen_12' 'pagie_1' 'vladislavleva_1' 'koza_3' 'keijzer_6' 'vladislavleva_8' 'korns_13' 'korns_14' 'korns_15')
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

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