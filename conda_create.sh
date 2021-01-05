#!/bin/bash

# Run script as follows:
# $ ./conda_create.sh

# Step 1: create conda env and activate env
# activate conda env from script:
#   stackoverflow.com/questions/55507519
conda create -n pdx python=3.7 pip setuptools wheel virtualenv --yes
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pdx

# Step 2: install cuda tools
conda install -c anaconda cudatoolkit=10.1 --yes

# Step 3: install openslide (the C library) with conda
conda install -c conda-forge openslide=3.4.1 --yes

# Note!
# You must activate the conda env before isntalling the remaining
# packages with requirement.txt
# $ conda activate pdx

# Step 4: install openslide-python with conda or pip
# conda install -c bioconda openslide-python=1.1.1 --yes
# pip install openslide-python

# Step 5: (manually) activate pdx
# conda activate pdx

# Step 6: (manually) activate create venv and install packages with Makefile
# make requirements
