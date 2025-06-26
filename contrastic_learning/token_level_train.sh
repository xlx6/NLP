#!/bin/bash
K_VALUES=(30 60 90)
ALPHA_VALUES=(0.2 0.5)

mkdir -p logs

for k in "${K_VALUES[@]}"; do
    for alpha in "${ALPHA_VALUES[@]}"; do
        echo "Running k=$k alpha=$alpha..."
        python imdb_deberta_tokenlevel_scl.py --k $k --alpha $alpha > "logs/k_${k}_alpha_${alpha}.log" 2>&1
        sleep 10
    done
done

echo "All experiments completed! Results saved in ./results/"