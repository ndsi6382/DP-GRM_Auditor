#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export OMP_NUM_THREADS=$(nproc --all)
export MKL_NUM_THREADS=$(nproc --all)
export OPENBLAS_NUM_THREADS=$(nproc --all)
export NUMEXPR_NUM_THREADS=$(nproc --all)
export TF_NUM_INTRAOP_THREADS=$(nproc --all)
export TF_NUM_INTEROP_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

source ~/miniconda3/etc/profile.d/conda.sh # Or path to where your conda is
conda activate privdpr

for g in 0 1; do
    for i in {0..399}; do
        nice -n 10 python3 PrivDPR_mem_saving.py --run_id ${i}_g${g} --epsilon 0.1 --dataset "Karate-Club" --data_path "../Data/Karate-Club/nx_adj${g}.pkl"
        if (( (i + 1) % 100 == 0 )); then
            wait
        fi
    done
    wait
done


for c in A B C D; do
    for g in 0 1; do
        for r in u d; do
            for i in {0..399}; do
                nice -n 10 python3 PrivDPR_mem_saving.py --run_id ${i}_g${g}${r} --epsilon 0.1 --dataset "Construction-${c}" --data_path "../Data/Construction-${c}/nx_adj${g}${r}.pkl" &
                if (( (i + 1) % 100 == 0 )); then
                    wait
                fi     
            done
            wait
        done
    done
done

for c in "Anybeat" "RT-LOLGOP"; do
    for g in 0 1; do
        for i in {0..399}; do
            nice -n 10 python3 PrivDPR_mem_saving.py --run_id ${i}_g${g} --epsilon 0.1 --dataset ${c} --data_path "../Data/${c}/nx_adj${g}.pkl" &
            if (( (i + 1) % 10 == 0 )); then
                wait
            fi
        done
        wait
    done
done

conda deactivate