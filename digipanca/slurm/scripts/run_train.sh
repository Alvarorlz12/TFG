#!/bin/bash

# Este script se encarga de lanzar un experimento de entrenamiento en el cluster
# de la UGR. Para ello, se le debe pasar el nombre del experimento y, opcionalmente,
# el nombre de la configuración a utilizar. Si no se especifica, se utilizará la
# configuración con nombre igual al del experimento.
if [ -z "$1" ]; then
    echo "Uso: $0 <nombre_experimento> [<nombre_configuración>]"
    exit 1
fi

EXPERIMENT_NAME=$1
CONFIG_NAME=${2:-$1}    # Si no se especifica, se utiliza el nombre del experimento

# Pasar el script modificado directamente a sbatch a través de stdin
sed -e "s/EXPERIMENT_NAME/${EXPERIMENT_NAME}/g" \
    -e "s/CONFIG_NAME/${CONFIG_NAME}/g" \
    slurm/scripts/train_template.sh | sbatch
