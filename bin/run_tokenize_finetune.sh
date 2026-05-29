#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j yes 
#$ -pe smp 16
##$ -l mem_free=32G
##$ -l h_rt=14:00:00
##$ -m ea
# ----------------------------------------------------------------------------------------
# Pre-process a corpus dataset before running fine-tuning.
# Performs train/test split, gene tokenization, token ranking.
# ----------------------------------------------------------------------------------------
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$PROJECT_ROOT"
TOTAL_CPUS=$(nproc --all)

# ----------------------------------------------------------------------------------------
# USER MODIFIED VARIABLES
CPUS=16  # Or: $((TOTAL_CPUS - 2)); HPC: $NSLOTS

MODALITY="Atlas"
MATRIX_FILE="Atlas_1pct_matrix.csv.gz"
MODEL_NAME="pc_atlas"
METADATA="Atlas_1pct_metadata.csv"
GENES="gene_names"
STAGE="finetune"
EMBED_DIM=1024          # Options: 512 1024 2048

INPUT_DIR="${PROJECT_ROOT}/data/${MODALITY}"
CACHE_DIR="${PROJECT_ROOT}/cache"
LABELS="${INPUT_DIR}/${METADATA}"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

DATA_MATRIX="${INPUT_DIR}/${MATRIX_FILE}"
GENE_NAMES_FILE="${INPUT_DIR}/${GENES}"
CACHE_PREFIX="${CACHE_DIR}/${STAGE}/${MODEL_NAME}" # _embed_${EMBED_DIM}"   # $(date +%Y%m%d_%H%M%S)"
BARCODES="${CACHE_DIR}/pretrain/${MODEL_NAME}/metadata/holdout_barcodes.txt"

# ----------------------------------------------------------------------------------------
# ENVIRONMENT SETUP

print_cpu_info() {
    echo "CPU: $(uname -m) | Detected ${TOTAL_CPUS} CPU cores - using ${CPUS}"
}

setup_environment() {
    echo "Activating virtual environment..."
    #conda activate pt
    #module load cuda/11.8
    #module load cuda/12.2
    #source venv/bin/activate
}

prepare_directories() {
    echo "Creating output and cache directories..."
    mkdir -p "${CACHE_DIR}"
    #mkdir -p "${OUTPUT}"
}

show_inputs_summary() {
    echo "================================"
    echo " MODEL:          $MODEL_NAME"
    echo " EMBED_DIM:      $EMBED_DIM"
    echo " DATA MATRIX:    $DATA_MATRIX"
    echo " CACHE PREFIX:   ${CACHE_PREFIX}*"
    echo "================================"
    echo "TRUE_LABELS=${LABELS}, BARCODES=${BARCODES}"
}

show_inputs_summary
print_cpu_info
prepare_directories
#setup_environment

# ----------------------------------------------------------------------------------------
# RUN THE SCRIPT

python -m src.preprocess.tokenize_finetune \
    "${DATA_MATRIX}" \
    "${GENE_NAMES_FILE}" \
    "${CACHE_PREFIX}" \
    "${LABELS}" \
    "${BARCODES}" \
    "${CONFIG_FILE}" \
    "${CPUS}"
