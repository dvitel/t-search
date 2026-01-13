#!/bin/bash 
config='koza_set1'

datasets=('r_1' 'r_2' 'keijzer_3' 'keijzer_4' 'keijzer_11' 'nguyen_12' 'pagie_1' 'vladislavleva_1' 'koza_3' 'keijzer_6' 'vladislavleva_8' 'korns_13' 'korns_14' 'korns_15')

for seed in {0..9}; do
    for dataset in "${datasets[@]}"; do
        echo "====================================================================="
        echo "Running t-search with dataset=$dataset, config=$config, seed=$seed"

        python -m t_search \
            --dataset $dataset \
            --config ./configs/$config.json \
            --output ./data/results.jsonlist \
            --device cuda \
            --dtype float32 \
            --seed $seed

        echo "Finished t-search with dataset=$dataset, config=$config, seed=$seed"
    done 
done