#!/bin/bash
# ========================================================================================
#$ -S /bin/bash
#$ -cwd
#$ -j yes
#$ -pe smp 2
#$ -q gpu.q
##$ -l mem_free=20G
##$ -l h_rt=05:00:00
##$ -m ea
# ========================================================================================
# Pre-train a model to learn general representations from a corpus dataset
# ========================================================================================
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$PROJECT_ROOT"
TOTAL_CPUS=$(nproc --all)

# ========================================================================================
# USER MODIFIED VARIABLES
CPUS=2  # Or: $((TOTAL_CPUS - 2)); Wynton: $NSLOTS

MODALITY="Xenium"
MATRIX_FILE="Xenium_AA_5pct_matrix.csv.gz"
MODEL_NAME="xenium_aa_5pct"
METADATA="Xenium_AA_5pct_metadata.csv"
GENES="gene_names_xenium.txt"
STAGE="pretrain"
EMBED_DIM=1024          # Options: 512 1024 2048


INPUT_DIR="${PROJECT_ROOT}/data/${MODALITY}"
CACHE_DIR="${PROJECT_ROOT}/cache"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

DATA_MATRIX="${INPUT_DIR}/${MATRIX_FILE}"
GENE_NAMES_FILE="${INPUT_DIR}/${GENES}"
CACHE_PREFIX="${CACHE_DIR}/${STAGE}/${MODEL_NAME}" # _embed_${EMBED_DIM}"   # $(date +%Y%m%d_%H%M%S)"
OUTPUT="$PROJECT_ROOT/model_weights/${STAGE}/${MODEL_NAME}/embed_${EMBED_DIM}/${MODEL_NAME}_${EMBED_DIM}_ranked_model"           # Script adds `.pt` to weights file


# ========================================================================================
# ENVIRONMENT SETUP

print_cpu_info() {
    echo "CPU: $(uname -m) | Detected ${TOTAL_CPUS} CPU cores - using ${CPUS}"
}

print_gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU(s):"
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null || nvidia-smi
    cuda_ver=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]\+\).*/\1/p' | head -n1)

    if [ -z "$cuda_ver" ] && command -v nvcc >/dev/null 2>&1; then
      cuda_ver=$(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -n1)
    fi
    [ -n "$cuda_ver" ] && echo "CUDA: $cuda_ver" || echo "CUDA: unknown"
  elif command -v rocm-smi >/dev/null 2>&1; then
    echo "GPU(s) (ROCm):"
    rocm-smi --showproductname --showvbios --showmeminfo vram
  else
    echo "GPU: none detected (no nvidia-smi/rocm-smi)"
  fi
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
    echo " EMBED_DIM:      $EMBED_DIM"
    echo " DATA MATRIX:    $DATA_MATRIX"
    echo " CACHE PREFIX:   ${CACHE_PREFIX}*"
    echo " OUTPUT DIR:     $OUTPUT"
    echo "================================"
    echo "BARCODES=${BARCODES}"
}

show_inputs_summary
print_cpu_info
print_gpu_info
prepare_directories
#setup_environment
#disable_huggingface_cache


# ========================================================================================
# RUN THE SCRIPT

#CUDA_LAUNCH_BLOCKING=1 
python -m src.training.pretrain \
    "${DATA_MATRIX}" \
    "${GENE_NAMES_FILE}" \
    "${CACHE_PREFIX}" \
    "${CONFIG_FILE}" \
    "${OUTPUT}" \
    "${CPUS}"


# ========================================================================================

