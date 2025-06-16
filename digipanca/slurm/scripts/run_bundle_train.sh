#!/bin/bash

if [ -z "$1" ]; then
    echo "Uso: $0 <config_name>"
    exit 1
fi

CONFIG_NAME=$1
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Pasar el script modificado directamente a sbatch a través de stdin
sed -e "s/CONFIG_NAME/${CONFIG_NAME}/g" \
    slurm/scripts/bundle_train_template.sh | sbatch --output=slurm/out/bundle_${CONFIG_NAME}_${TIMESTAMP}_%j.out