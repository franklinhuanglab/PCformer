#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j yes
#$ -pe smp 16
##$ -l mem_free=30G
##$ -l h_rt=12:00:00
##$ -m ea
# ----------------------------------------------------------------------------------------
# Pre-process a corpus dataset before running pre-training.
# Performs train/test split, gene tokenization, token ranking.
# ----------------------------------------------------------------------------------------
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$PROJECT_ROOT"
TOTAL_CPUS=$(nproc --all)

# ----------------------------------------------------------------------------------------
# USER MODIFIED VARIABLES
CPUS=16  # Or: $((TOTAL_CPUS - 2)); HPC: $NSLOTS

MODALITY="Xenium"
MATRIX_FILE="Xenium_AA_5pct_matrix.csv.gz"
MODEL_NAME="xenium_aa_5pct"
METADATA="Xenium_AA_5pct_metadata.csv"
GENES="gene_names_xenium.txt"
STAGE="pretrain"


INPUT_DIR="${PROJECT_ROOT}/data/${MODALITY}"
CACHE_DIR="${PROJECT_ROOT}/cache"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

DATA_MATRIX="${INPUT_DIR}/${MATRIX_FILE}"
GENE_NAMES_FILE="${INPUT_DIR}/${GENES}"
CACHE_PREFIX="${CACHE_DIR}/${STAGE}/${MODEL_NAME}"   # $(date +%Y%m%d_%H%M%S)"


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

disable_huggingface_cache() {
    echo "Disabling Hugging Face datasets cache..."
    # Choose only one of these
    export HF_DATASETS_CACHE=None
    export HF_DATASETS_CACHE=""
}

prepare_directories() {
    echo "Creating output and cache directories..."
    mkdir -p "${CACHE_DIR}"
    #mkdir -p "${OUTPUT_DIR}"
}

show_inputs_summary() {
    echo "================================"
    echo " MODEL:          $MODEL_NAME"
    echo " DATA MATRIX:    $DATA_MATRIX"
    echo " CACHE PREFIX:   ${CACHE_PREFIX}*"
    echo "================================"
    echo "BARCODES=${BARCODES}"
}

show_inputs_summary
print_cpu_info
#print_gpu_info
prepare_directories
#setup_environment
#disable_huggingface_cache


# ----------------------------------------------------------------------------------------
# RUN THE SCRIPT

python -m src.preprocess.tokenize_pretrain \
    "${DATA_MATRIX}" \
    "${GENE_NAMES_FILE}" \
    "${CACHE_PREFIX}" \
    "${CONFIG_FILE}" \
    "${CPUS}"

