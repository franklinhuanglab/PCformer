#!/bin/bash
# Run this file on instance start. Output in /var/log/onstart.log
# https://pytorch.org/get-started/locally/

pip install pandas
pip install datasets
pip install tensorboard
pip install transformers
pip install anndata
pip install scanpy
pip install scikit-learn
pip install matplotlib
pip install seaborn

# Free up space: delete cache and trash
#rm -rf ~/.cache/huggingface/datasets/*
#rm -rf root/.local/share/Trash/files/*
