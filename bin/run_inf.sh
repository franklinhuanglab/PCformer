#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -q gpu.q
##$ -l mem_free=50G
##$ -l h_rt=144:00:00
##$ -m ea
# ----------------------------------------------------------------------------------------
# Use the fine-tuned model to predict cell types in a hold-out or new dataset
# ----------------------------------------------------------------------------------------
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$PROJECT_ROOT"
TOTAL_CPUS=$(nproc --all)

# ----------------------------------------------------------------------------------------
# USER MODIFIED VARIABLES
CPUS=8  # Or: $((TOTAL_CPUS - 2)); HPC: $NSLOTS

MODALITY="Atlas"                          # MODIFY. Where the input files live in `data`
MATRIX_FILE="Atlas_1pct_matrix.csv.gz"    # MODIFY. Input expression matrix
MODEL_NAME="pc_atlas"                     # MODIFY. Name of the fine-tuned model
METADATA="Atlas_1pct_metadata.csv"        # Model labels, not from new inf dataset
GENES="gene_names"                        # Gene vocabulary
STAGE="inference"
EMBED_DIM=1024                            # Options: 512 1024 2048

INPUT_DIR="${PROJECT_ROOT}/data/${MODALITY}"
CACHE_DIR="${PROJECT_ROOT}/cache"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

LABELS="${INPUT_DIR}/${METADATA}"
TRUE_LABELS="${INPUT_DIR}/${METADATA}"    # MODIFY. Options: "${LABELS}" or "NULL"
#BARCODES="NULL"                          # MODIFY. Cache hold-out barcodes or "NULL"
BARCODES="${CACHE_DIR}/pretrain/${MODEL_NAME}/metadata/holdout_barcodes.txt"

GENE_NAMES_FILE="${INPUT_DIR}/${GENES}"
MODEL="${PROJECT_ROOT}/model_weights/finetune/${MODEL_NAME}/embed_${EMBED_DIM}/${MODEL_NAME}_${EMBED_DIM}_finetuned.pt"

DATA_MATRIX="${INPUT_DIR}/${MATRIX_FILE}"
DATASET_NAME=$(basename "$DATA_MATRIX" ".csv.gz")

CACHE_PREFIX="${CACHE_DIR}/${STAGE}/${MODEL_NAME}/embed_${EMBED_DIM}"   # $(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PROJECT_ROOT}/results/${MODEL_NAME}/${DATASET_NAME}_${EMBED_DIM}_${STAGE}"
OUTPUT_PREFIX="${OUTPUT_DIR}/${DATASET_NAME}_${STAGE}"

# ----------------------------------------------------------------------------------------
echo "${TRUE_LABELS}, ${BARCODES}"

# CPU/GPU Setup
print_cpu_info() {
    echo "Detected ${TOTAL_CPUS} CPU(s) — using ${CPUS}"
}

# GPU Setup
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

setup_environment() {
    echo "Activating conda environment..."
    # conda activate pt
    # module load cuda/11.8
    # module load cuda/12.2
    # source venv/bin/activate
}

prepare_directories() {
    echo "Creating output and cache directories..."
    #mkdir -p "${CACHE_DIR}"
    mkdir -p "${OUTPUT_DIR}"
}

show_inputs_summary() {
    echo "================================"
    echo " MODEL:          $MODEL_NAME"
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
prepare_directories

# ----------------------------------------------------------------------------------------
# RUN THE SCRIPT

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
