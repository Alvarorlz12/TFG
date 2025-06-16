#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Uso: $0 <nombre_experimento> <nombre_configuración> <num_epochs> <cola>"
    exit 1
fi

EXPERIMENT_NAME=$1
CONFIG_NAME=$2
EPOCHS=$3
PARTITION=$4

# Incluir el parámetro epochs si es distinto de 0
if [ "$EPOCHS" -eq 0 ]; then
    EPOCHS_LINE=""
else
    EPOCHS_LINE="--num_epochs $EPOCHS"
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)    # Fecha y hora actual

# Pasar el script modificado directamente a sbatch a través de stdin
sed -e "s/EXPERIMENT_NAME/${EXPERIMENT_NAME}/g" \
    -e "s/CONFIG_NAME/${CONFIG_NAME}/g" \
    -e "s/PART/${PARTITION}/g" \
    -e "s|EPOCHS_LINE|${EPOCHS_LINE}|g" \
    slurm/scripts/fit_template.sh | sbatch --output=slurm/out/${EXPERIMENT_NAME}_${TIMESTAMP}_%j.out
