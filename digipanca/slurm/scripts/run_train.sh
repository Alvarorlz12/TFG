#!/bin/bash

# Este script se encarga de lanzar un experimento de entrenamiento en el cluster
# de la UGR. Para ello, se le debe pasar el nombre del experimento y, opcionalmente,
# el nombre de la configuración a utilizar. Si no se especifica, se utilizará la
# configuración con nombre igual al del experimento.
# Si no se especifica la cola, se usará "dios" por defecto.
if [ -z "$1" ]; then
    echo "Uso: $0 <nombre_experimento> [<nombre_configuración>] [<cola>]"
    exit 1
fi

EXPERIMENT_NAME=$1
CONFIG_NAME=${2:-$1}    # Si no se especifica, se utiliza el nombre del experimento
PARTITION=${3:-dios}    # Si no se especifica, se usa "dios"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)    # Fecha y hora actual

# Pasar el script modificado directamente a sbatch a través de stdin
sed -e "s/EXPERIMENT_NAME/${EXPERIMENT_NAME}/g" \
    -e "s/CONFIG_NAME/${CONFIG_NAME}/g" \
    -e "s/PART/${PARTITION}/g" \
    slurm/scripts/train_template.sh | sbatch --output=slurm/out/${EXPERIMENT_NAME}_${TIMESTAMP}_%j.out
