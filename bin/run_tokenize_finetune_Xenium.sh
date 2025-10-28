#!/bin/bash
# ========================================================================================
#$ -S /bin/bash
#$ -cwd
#$ -j yes 
#$ -pe smp 16
##$ -l mem_free=32G
##$ -l h_rt=14:00:00
##$ -m ea
# ========================================================================================
# Pre-process the a corpus dataset for fine-tuning -
# train/test split, gene tokenization, token ranking
# ========================================================================================
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$PROJECT_ROOT"

# USER MODIFIED VARIABLES
# -----------------------
TOTAL_CPUS=$(nproc --all)
CPUS=16  # Or: $((TOTAL_CPUS - 2)); Wynton: $NSLOTS

MODALITY="Xenium"
MATRIX_FILE="Xenium_AA_5pct_matrix.csv.gz"
MODEL_NAME="xenium_aa_5pct"
METADATA="Xenium_AA_5pct_metadata.csv"
GENES="gene_names_xenium.txt"
STAGE="finetune"
EMBED_DIM=1024          # Options: 512 1024 2048


INPUT_DIR="${PROJECT_ROOT}/data/${MODALITY}"
CACHE_DIR="${PROJECT_ROOT}/cache"
LABELS="${INPUT_DIR}/${METADATA}"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

DATA_MATRIX="${INPUT_DIR}/${MATRIX_FILE}"
GENE_NAMES_FILE="${INPUT_DIR}/${GENES}"
CACHE_PREFIX="${CACHE_DIR}/${STAGE}/${MODEL_NAME}" # _embed_${EMBED_DIM}"   # $(date +%Y%m%d_%H%M%S)"


# ========================================================================================
# ENVIRONMENT SETUP

print_cpu_info() {
    echo "CPU: $(uname -m) | Detected ${TOTAL_CPUS} CPU cores - using ${CPUS}"
}

print_gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU(s):"
    # Query only fields supported broadly
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null || nvidia-smi
    cuda_ver=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]\+\).*/\1/p' | head -n1)
    # Fallback to nvcc if CUDA toolkit is installed
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
    # Choose to activate conda or venv, and cuda version
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
#print_gpu_info
prepare_directories
#setup_environment
#disable_huggingface_cache

# ========================================================================================
# RUN THE SCRIPT

python -m src.preprocess.tokenize_finetune \
    "${DATA_MATRIX}" \
    "${GENE_NAMES_FILE}" \
    "${CACHE_PREFIX}" \
    "${LABELS}" \
    "${CONFIG_FILE}" \
    "${CPUS}"

# ========================================================================================

