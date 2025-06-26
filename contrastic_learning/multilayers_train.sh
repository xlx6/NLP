#!/bin/bash
k=50
alpha=0.2
epochs=3

declare -a levels=("token" "sentence")
declare -a layers_combinations=("0 1 2 3" "4 5 6 7" "8 9 10 11" "5 7 9 11" "2 5 8 11")

mkdir -p ./logs

for level in "${levels[@]}"
do
    for layers in "${layers_combinations[@]}"
    do
        log_file="./logs/layers_${layers// /_}_level_${level}_k${fixed_k}_alpha${fixed_alpha}.log"
        
        echo "Running experiment with layers $layers, level $level, k=$fixed_k, alpha=$fixed_alpha" | tee $log_file
        python imdb_deberta_multilayers_contrastic.py \
            --k $k \
            --alpha $alpha \
            --layers $layers \
            --epochs $epochs \
            --level $level >> $log_file 2>&1
    done
done
