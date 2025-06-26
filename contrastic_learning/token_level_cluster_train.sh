#!/bin/bash
K_VALUES=(90)
ALPHA_VALUES=(0.25)
N_CLUSTERS=(5 10 15)

mkdir -p logs

for k in "${K_VALUES[@]}"; do
    for alpha in "${ALPHA_VALUES[@]}"; do
        for nc in "${N_CLUSTERS[@]}"; do
            echo "Running k=$k alpha=$alpha n_clusters=$nc..."
            python imdb_deberta_tokenlevel_scl.py --k $k --alpha $alpha --nc $nc > "logs/k_${k}_alpha_${alpha}_nc_${nc}.log" 2>&1
            sleep 10
        done
    done
done

echo "All experiments completed! Results saved in ./results/"