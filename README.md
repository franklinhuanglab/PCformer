**PCformer**
=====================
`Transformer model for the annotation of prostate cell types from ranked-value single-cell and single-nucleus RNA, ATAC, Xenium, and Visium data.` 

Prostate cancer (PCa) shows strong heterogeneity in gene expression, making tumor cell annotation difficult. Current single-cell annotation methods rely on non-standardized approaches, often leading to inconsistent results. To address this, we developed a deep learning transformer model that standardizes the annotation of PCa cells. The architecture builds on a single-cell RNA foundation model trained on 560,492 cells and 13,745 genes from the PCa Atlas. This foundation model learns cell type specific gene expression patterns, which are then fine-tuned for the classification of prostate cell in new data.

**DATASETS**
--------
---

**> Prostate single-cell RNA-seq matrix**

Format

|        | GeneA | GeneB | GeneC | GeneD | GeneE |
|--------|-------|-------|-------|-------|-------|
| Cell_1 | float | float | float | float | float |
| Cell_2 | float | float | float | float | float |
| Cell_3 | float | float | float | float | float |

**> Metadata**
|         |           |      | Cell Type Classes |          |           |
|---------|-----------|--------|---------------|------------|-----------|
| Basal   | Bcell     | Club   | Endothelial   | Fibroblast | Mast_cell | 
| Myeloid | Non_Tumor | Plasma | Smooth_Muscle | Tcell      | Tumor     |
|  |  |  |  |  |  |  

Format

| CellName | ID |
|----------|-------|
| Cell_1   | Tcell |
| Cell_2   | Club  |
| Cell_3   | Tumor |
| Cell_4   | Basal |
| Cell_5   | Tumor |



**HOW TO RUN INFERENCE**
------------
____________

All the job submission scripts can be found under the `bin` directory.
```
.
├── bin/                                -> Shell scripts
│   ├── run_tokenize_pretrain.sh
│   ├── run_tokenize_finetune.sh
│   ├── run_pretrain.sh
│   ├── run_finetune.sh
│   ├── run_inf.sh                      -> Inference submission script
```

Make a copy of `run_inf.sh` and use a filename suffix to describe this new version of the file (e.g., `run_inf_Xenium.sh`).

```
Modify the user variables:
o CPUS=N                                 -> If planning to use GPU, CPU can be set to a low value
o MODALITY="Xenium"                      -> `data` subdirectory where the input data lives
o MATRIX_FILE="Xenium_AA_matrix.csv.gz"  -> Gene expression matrix; main input file
o MODEL_NAME="xenium_aa"                 -> Model name that the user chose during training             
o METADATA="Xenium_AA_metadata.csv"      -> Metadata used for training; [ barcode    Label ]
o GENES="gene_names_xenium.txt"          -> Vocabulary gene names for the tokenizer
o STAGE="inference"                      -> Stages: <pretrain> <finetune> <inference>        
o EMBED_DIM=1024                         -> Embedding size (1024 dimensions). Other options: 512, 2048

o LABELS="${INPUT_DIR}/${METADATA}"      -> The set of training labels from the metadata
o TRUE_LABELS="${LABELS}"                -> if running on hold-out dataset, use "${LABELS}" 
                                            if using new dataset, specify the metadata directory
                                            if labels unavailable, use "NULL"
o BARCODES="${CACHE_DIR}/finetune/${MODEL_NAME}/metadata/inference_barcodes.txt" 
                                         -> if hold-out dataset, use cache finetune inference_barcodes.txt dir
                                            else, use "NULL" 
```

Run in the terminal: `bash run_inf.sh`


**WORKFLOW**
------------
____________

![INK](images/transformer_schematic_v3.png)


**Pre-processing the Corpus Data for Pre-training**
---------------------------------------------------
______________________________________________

Script: `tokenize_pretrain.py`

The initial step involves splitting the data into two sets: 1) A training set composed of 90% of the cells in the dataset to train the model, and 2) A validation set composed of 10% of the dataset used to evaluate the model's performance during training.

Each set is individually passed to a tokenizer that first removes gene expressions with zero value and then ranks the genes by importance. The gene names are then tokenized into unique numeric IDs.

Input:
 - **< DATA_MATRIX >** (_csv_): Contains the cell gene expression matrix. The script supports file formats [.csv, .csv.gz] and delimiters [comma, tab].  
    _Example: data/scRNA/Atlas_Matrix.csv.gz_
 - **< GENE_NAMES_FILE >** (_list_): List of gene names included in the expression matrix; the vocabulary.  
    _Example: data/scRNA/gene_names.txt_
 - **< CACHE_PREFIX >**  (_str_): Directory prefix for caching tokenized input data.  
    _Example: cache/inference/pc_atlas/Atlas_Matrix_1024_
 - **< CONFIG_FILE >** (_yaml_): Configuration file used to pass parameters.  
    _Example: config.yaml_
 - **< CPUS >** (_int_): Number of CPUs for parallelization.  
    _Example: 16_

Output:
```
.
├── metadata
│   ├── test_barcodes.txt  -> Test set barcodes
│   ├── train_barcodes.txt -> Train set barcodes
├── train
│   ├── data-*.arrow      -> [int, int, int, ...]
│   ├── dataset_info.json
│   └── state.json
├── test
│   ├── data-*.arrow      -> [int, int, int, ...]
│   ├── dataset_info.json
│   └── state.json
```


**Pre-training the Foundation Model**
------------------------------------
_____________________________________

Script: `pretrain.py`

Pre-training step where the ranked tokenized genes and their corresponding gene expression profiles are used to learn generalized representations, using the train and test datasets. By attending to all tokens (i.e., genes) within each cell, the transformer model learns contextual relationships, capturing gene dependencies and interactions. Thus, the model learns representations from the structure of the data in a self-supervised manner.

Input:
 - **< DATA_MATRIX >** (_csv_): Contains the cell gene expression matrix. The script supports file formats [.csv, .csv.gz] and delimiters [comma, tab].  
    _Example: data/scRNA/Atlas_Matrix.csv.gz_
 - **< GENE_NAMES_FILE >** (_list_): List of gene names included in the expression matrix; the vocabulary.  
    _Example: data/scRNA/gene_names.txt_
 - **< CACHE_PREFIX >**  (_str_): Directory prefix where the cached tokens are located.  
    _Example: cache/finetune/pc_atlas/embed_1024/_
 - **< OUTPUT >** (_text_): Path prefix for the .pt model checkpoint file.
    _Example: model_weights/pretrain/pc_atlas/embed_1024/pc_atlas_1024_
 - **< CONFIG_FILE >** (_yaml_): Configuration file used to pass the hyperparameters.  
    _Example: config.yaml_
 - **< CPUS >** (_int_): Number of CPUs for parallelization.  
    _Example: 16_

Output:
 - **< OUTPUT >.pt** (PyTorch): A PyTorch file containing the foundation model weights.


**Pre-processing the Corpus Data for Fine-tuning**
--------------------------------------------------
___________________________________

Script: `tokenize_finetune.py`

Data preprocessing step preceding the `finetune.py` script. The dataset is split into 80% training, 10% testing, and 10% inference cells. The train and test sets are processed to remove genes with zero expression, rank genes by importance, and tokenize gene names.

Input:
 - **< DATA_MATRIX >** (_csv_): File containing the cell gene expression matrix. The script supports file formats [.csv, .csv.gz] and delimiters [comma, tab].  
    _Example: data/scRNA/Atlas_Matrix.csv.gz_
 - **< GENE_NAMES_FILE >** (_list_): List of gene names included in the expression matrix; the vocabulary.  
    _Example: data/scRNA/gene_names.txt_
 - **< CACHE_PREFIX >**  (_str_): Directory prefix for caching tokenized input data.  
    _Example: cache/inference/pc_atlas/Atlas_Matrix_1024_
 - **< LABELS >** (_csv_): Metadata containing the ground-truth class labels for each barcode; used to extract the 12 cell type classes. Compatible with delimiters [comma, tab].  
    _Example: data/scRNA/Atlas_Metadata.csv_
 - **< CONFIG_FILE >** (_yaml_): Configuration file used to pass hyperparameters.  
    _Example: config.yaml_
 - **< CPUS >** (_int_): Number of CPUs for parallelization.  
    _Example: 16_

Output:  
```
.
├── metadata
│   ├── inference_barcodes.txt -> Hold-out set barcodes
│   ├── test_barcodes.txt      -> Test set barcodes
│   ├── train_barcodes.txt     -> Train set barcodes
├── test
│   ├── data-*.arrow       -> [int, int, int, ...]
│   ├── dataset_info.json
│   └── state.json
├── train
│   ├── data-*.arrow       -> [int, int, int, ...]
│   ├── dataset_info.json
│   └── state.json
├── 
```


**Fine-tuning Task for Cell Type Classification**
------------------------------------------------
________________________________________________

Script: `finetune.py` 

Finetuning task for cell type classification using the tokenize-finetune tokens and cell0type labels. The model is trained on 80% of the dataset and validated on a set of 10% of the cells. The remaining 10% is held-out for inference. Here, barcode ground-truth labels are used to allow the model to learn representations for each cell type.

Input:
 - **< DATA_MATRIX >** (_csv_): File containing the cell gene expression matrix. The script supports file formats [.csv, .csv.gz] and delimiters [comma, tab].  
    _Example: data/scRNA/Atlas_Matrix.csv.gz_
 - **< GENE_NAMES_FILE >** (_list_): List of gene names included in the expression matrix; the vocabulary.  
    _Example: data/scRNA/gene_names.txt_
 - **< CACHE_PREFIX >**  (_str_): Directory prefix where the cached tokens are located.  
    _Example: cache/finetune/pc_atlas/embed_1024/_
 - **< OUTPUT >** (_PyTorch_): Path to the .pt model checkpoint file; fine-tuned classification model.
    _Example: model_weights/finetune/pc_atlas/embed_1024/pc_atlas_1024_finetuned.pt_
 - **< LABELS >** (_csv_): Metadata containing the ground-truth class labels for each barcode; used to extract the 12 cell type classes. Compatible with delimiters [comma, tab].  
    _Example: data/scRNA/Atlas_Metadata.csv_
 - **< CONFIG_FILE >** (_yaml_): Configuration file used to pass hyperparameters.  
    _Example: config.yaml_
 - **< OUTPUT_PREFIX >** (_str_): Directory and prefix for output results.  
    _Example: results/pc_atlas/Atlas_Matrix_1024_inference/Atlas_Matrix_1024_inference*_
 - **< CPUS >** (_int_): Number of CPUs for parallelization.  
    _Example: 16_

Output:
 - **< OUTPUT >** (PyTorch): The PyTorch file containing the finetuned model's weights.


**Inference**
-------------
_____________

Script: `inference.py`

Inference script to test the generalizability of the model on unseen data. Used to make predictions on a hold-out dataset, representing 10% of the corpus cells, or on new datasets. The script will adjust for differences in the set of genes from the new dataset and apply padding to match the vocabulary size.


Input:
 - **< DATA_MATRIX >** (_csv_): File containing the cell gene expression matrix. The script supports file formats [.csv, .csv.gz] and delimiters [comma, tab].  
    _Example: data/scRNA/Atlas_Matrix.csv.gz_
 - **< GENE_NAMES_FILE >** (_list_): List of gene names included in the expression matrix; the vocabulary.  
    _Example: data/scRNA/gene_names.txt_
 - **< CACHE_PREFIX >**  (_str_): Directory prefix for caching tokenized input data.  
    _Example: cache/inference/pc_atlas/Atlas_Matrix_1024_
 - **< MODEL >** (_PyTorch_): Path to the .pt model checkpoint file; fine-tuned classification model.
    _Example: model_weights/finetune/pc_atlas/embed_1024/pc_atlas_1024_finetuned.pt_
 - **< LABELS >** (_csv_): Metadata containing the ground-truth class labels per barcode; used to extract the 12 cell type classes. Compatible with delimiters [comma, tab].  
    _Example: data/scRNA/Atlas_Metadata.csv_
 - **< TRUE_LABELS >** (_list_ or _NULL_): Optional. Metadata containing the ground-truth labels per barcode. Use "NULL" if unavailable. Compatible with delimiters [comma, tab].  
    _Example: data/scRNA/Atlas_Metadata.csv_
 - **< BARCODES >** (_list_ or _NULL_): Optional. List of hold-out barcodes. Use "NULL" if not used.  
    _Example: cache/finetune/pc_atlas/embed_1024/metadata/inference_barcodes.txt_
 - **< CONFIG_FILE >** (_yaml_): Configuration file used to pass parameters.  
    _Example: config.yaml_
 - **< OUTPUT_PREFIX >** (_str_): Directory and prefix for output results.  
    _Example: results/pc_atlas/Atlas_Matrix_1024_inference/Atlas_Matrix_1024_inference*_
 - **< CPUS >** (_int_): Number of CPUs for parallelization.  
    _Example: 16_

Output:
 - **< OUTPUT_PREFIX >.csv** (_csv_): The predicted labels for each cell in the input matrix. CellName,PredictedLabel,Confidence

**Project Structure and Data Transfer**
------------------
---

The updated source code will live in [https://github.com/franklinhuanglab/PCformer.git](https://github.com/franklinhuanglab/PCformer.git). 
Each user will clone this repo into a rented Vast.ai machine each time they plan to use the transformer architecture.

# <pre lang="markdown">
## Directory Tree
```
.
├── bin/                                -> Shell scripts
│   ├── run_tokenize_pretrain.sh        -> Tokenize dataset for pretraining
│   ├── run_tokenize_finetune.sh        -> Tokenize dataset for fine-tuning
│   ├── run_pretrain.sh                 -> Launch pretraining run
│   ├── run_finetune.sh                 -> Launch fine-tuning run
│   ├── run_inf.sh                      -> General inference script
├── data/                               -> DATA DIR TO BE TRANSFERRED INDIVIDUALLY
│   ├── Xenium
│   │   ├── <GENES>.txt
│   │   ├── <MATRIX>.csv.gz
│   │   └── <METADATA>.tsv
├── cache/                              -> TOKENIZED DATA DIRS TO BE TRANSFERRED INDIVIDUALLY
│   ├── pretrain/
│   │   └── <MODEL>/
│   │       ├── metadata/               -> Train/test/inference barcode lists
│   │       ├── train/                  -> Tokenized train dataset
│   │       └── test/                   -> Tokenized test dataset
│   ├── finetune/
│   │   └── <MODEL>/
│   │       ├── metadata/               -> Train/test/inference barcode lists
│   │       ├── train/                  -> Tokenized train dataset
│   │       └── test/                   -> Tokenized test dataset
├── model_weights/                      -> MODEL WEIGHTS AND OUTPUTS TO BE TRANSFERRED INDIVIDUALLY
│   ├── pretrain/
│   │   └── <MODEL>/embed_<DIM>/        -> Pretrained models
│   │       ├── *.pt
│   ├── finetune/
│   │    └── <MODEL>/embed_<DIM>/       -> Fine-tuned models
│   │       └── *.pt
├── results/                            -> INFERENCE OUTPUTS WILL LIVE HERE
│   ├── <MODEL>
│   │   └── <DIM>
│   │       ├── inference_<MODEL>.csv
│   │       └── inference_<MODEL>_classification_report.txt
├── runs/                               -> JOB RUN LOGS SPECIFIC TO EACH USER
│   └── *.log
├── src/                                -> Python source code; main project scripts
│   ├── __init__.py
│   ├── model/                          -> Model definitions and configs
│   │   ├── __init__.py
│   │   ├── atlas_model_rank_based.py
│   │   ├── configuration_atlas_model_rank_based.py
│   │   └── genes.py
│   ├── preprocess/                     -> Tokenization and data loading
│   │   ├── __init__.py
│   │   ├── data_utils.py
│   │   ├── gene_expression_datasets.py
│   │   ├── tokenize_dataset.py
│   │   ├── tokenize_finetune.py
│   │   └── tokenizer.py
│   ├── training/                       -> Training and inference scripts
│   │   ├── __init__.py
│   │   ├── finetune.py
│   │   ├── inference.py
│   │   └── pretrain.py
├── tools/                              -> Utility scripts
│   ├── extract_attn_importance.py      -> In development for GRN
│   ├── extract_h5ad_labels.py          -> Extract annotations labels from an h5ad file
│   ├── h5_to_csv.R                     -> Extract a matrix file from h5/h5ad (R)
│   ├── matrix_to_npy.py                -> Convert a matrix to split col/index/matrix Numpy files
│   ├── run_extract_attn_importance.py  -> In development for GRN
│   ├── create_csv_matrix_from_h5ad.py  -> Extract a matrix file from h5/h5ad
│   └── subset_csv_matrix.py            -> Subset a matrix to a fraction of the rows
├── config.yaml
├── onstart.sh
├── README.md
├── requirements.txt
```

# </pre>

---


<p align="center"> 
<a href="https://www.linux.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> </a> 
<a href="https://www.gnu.org/software/bash/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/gnu_bash/gnu_bash-icon.svg" alt="bash" width="40" height="40"/> </a>  
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> 
<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> 
<a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a> 
<a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> </p>