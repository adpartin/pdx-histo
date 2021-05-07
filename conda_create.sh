#!/bin/bash

# Run script as follows:
# $ ./conda_create.sh

# TF-cudnn-cudatoolkit configs
# https://www.tensorflow.org/install/source#gpu

# The installation below is for lambda machines.
# For lamina, see conda_create_on_lamina.sh

# Step 1: create conda env and activate env
# activate conda env from script:
#   stackoverflow.com/questions/55507519
conda create -n pdx python=3.7 pip setuptools wheel virtualenv --yes
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pdx

# Step 2: install cudatoolkit tools
conda install -c anaconda cudatoolkit=10.1 --yes

# Step 3: install cudnn tools
conda install -c anaconda cudnn=7.6 --yes

# Step 4: install openslide (the C library) with conda
conda install -c conda-forge openslide=3.4.1 --yes

# Note!
# You must activate the conda env before isntalling the remaining
# packages with requirement.txt
# $ conda activate pdx

# Step 5: install openslide-python with conda or pip
# conda install -c bioconda openslide-python=1.1.1 --yes
# pip install openslide-python

# Step 6: (manually) create venv and install packages with Makefile with one of the following:
# make dev-venv
# make dev-venv_tf23
# make dev-venv_tf24
