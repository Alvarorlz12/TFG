#!/bin/bash

if [ -z "$1" ]; then
    echo "Uso: $0 <nombre_experimento>"
    exit 1
fi

EXPERIMENT_NAME=$1

# Pasar el script modificado directamente a sbatch a través de stdin
sed "s/EXPERIMENT_NAME/${EXPERIMENT_NAME}/g" slurm/scripts/train_template.sh | sbatch
