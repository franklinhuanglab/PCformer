#!/bin/bash
# ========================================================================================
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
##$ -l mem_free=50G
##$ -l h_rt=144:00:00
##$ -m ea
# =============================================================================
# Use the fine-tuned model to predict cell types in a hold-out or new dataset
# =============================================================================
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$PROJECT_ROOT"

# USER MODIFIED VARIABLES
TOTAL_CPUS=$(nproc --all)
CPUS=8  # Or: $((TOTAL_CPUS - 2)); Wynton: $NSLOTS

MODALITY="Xenium"
MATRIX_FILE="Xenium_AA_5pct_matrix.csv.gz"
MODEL_NAME="xenium_aa_5pct"
METADATA="Xenium_AA_5pct_metadata.csv"
GENES="gene_names_xenium.txt"
STAGE="inference"
EMBED_DIM=1024          # Options: 512 1024 2048

INPUT_DIR="${PROJECT_ROOT}/data/${MODALITY}"
CACHE_DIR="${PROJECT_ROOT}/cache"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

LABELS="${INPUT_DIR}/${METADATA}"
TRUE_LABELS="${LABELS}" # Options: "${LABELS}" "NULL"
BARCODES="${CACHE_DIR}/finetune/${MODEL_NAME}/metadata/inference_barcodes.txt" # "NULL" 

echo "${TRUE_LABELS}, ${BARCODES}"

GENE_NAMES_FILE="${INPUT_DIR}/${GENES}"
MODEL="${PROJECT_ROOT}/model_weights/finetune/${MODEL_NAME}_embed_${EMBED_DIM}/${MODEL_NAME}_${EMBED_DIM}_finetuned.pt"

DATA_MATRIX="${INPUT_DIR}/${MATRIX_FILE}"
DATASET_NAME=$(basename "$DATA_MATRIX" ".csv.gz")

CACHE_PREFIX="${CACHE_DIR}/finetune/${MODEL_NAME}_embed_${EMBED_DIM}"   # $(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PROJECT_ROOT}/results/${MODEL_NAME}/${DATASET_NAME}_${EMBED_DIM}_${STAGE}"
OUTPUT_PREFIX="${OUTPUT_DIR}/${STAGE}_${DATASET_NAME}"


# CPU Setup
print_cpu_info() {
    TOTAL_CPUS=$(nproc --all)
    CPUS=16  # Or: $((TOTAL_CPUS - 2)); Or: $NSLOTS
    echo "Detected ${TOTAL_CPUS} CPU(s) — using ${CPUS}"
}
# GPU Setup (minimal)
print_gpu_info() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        CNT=$(nvidia-smi -L | wc -l)
        CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version[^0-9]*\([0-9.]\+\).*/\1/p;T;q')
        echo "Detected ${CNT} NVIDIA GPU(s) — CUDA ${CUDA}"
    elif command -v rocm-smi >/dev/null 2>&1; then
        CNT=$(rocm-smi -i | grep -c '^GPU\[')
        ROCM=$(rocm-smi --showdriverversion 2>/dev/null | awk -F': ' 'NR==1{print $2}')
        echo "Detected ${CNT} AMD GPU(s) — ROCm ${ROCM}"
    else
        echo "No GPU detected (nvidia-smi/rocm-smi not found)"
    fi
}

# ========================================================================================
setup_environment() {
    echo "Activating conda environment..."
    # conda activate pt
    # module load cuda/11.8
    # module load cuda/12.2
    # source venv/bin/activate
}

disable_huggingface_cache() {
    echo "Disabling Hugging Face datasets cache..."
    export HF_DATASETS_CACHE=None
    export HF_DATASETS_CACHE=""
}

prepare_directories() {
    echo "Creating output and cache directories..."
    mkdir -p "${CACHE_DIR}"
    mkdir -p "${OUTPUT_DIR}"
}

show_inputs_summary() {
    echo "================================"
    echo " MODEL:          $MODEL"
    echo " EMBED_DIM:      $EMBED_DIM"
    echo " DATA MATRIX:    $DATA_MATRIX"
    echo " CACHE PREFIX:   $CACHE_PREFIX"
    echo " OUTPUT DIR:     $OUTPUT_DIR"
    echo "================================"
    echo "TRUE_LABELS=${TRUE_LABELS}, BARCODES=${BARCODES}"
}

show_inputs_summary
print_cpu_info
print_gpu_info
setup_environment
#disable_huggingface_cache
prepare_directories


# ========================================================================================
# RUN THE SCRIPT

# Forces CUDA to run synchronously for easier debugging
# CUDA_LAUNCH_BLOCKING=1 
python -m src.training.inference \
    "${DATA_MATRIX}" \
    "${GENE_NAMES_FILE}" \
    "${CACHE_PREFIX}" \
    "${MODEL}" \
    "${LABELS}" \
    "${TRUE_LABELS}" \
    "${BARCODES}" \
    "${CONFIG_FILE}" \
    "${OUTPUT_PREFIX}" \
    "${CPUS}"


# ========================================================================================
