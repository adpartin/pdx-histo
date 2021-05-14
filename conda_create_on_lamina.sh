#!/bin/bash

# Run script as follows:
# $ ./conda_create_on_lamina.sh

# Create env by running the line below, and then run this script.
# conda create -n pdx_lamina python=3.7 pip --yes

conda install -c anaconda tensorflow-gpu=2.4.1 --yes
conda install -c anaconda flake8 --yes

conda install -c conda-forge ipdb=0.13.4 --yes
conda install -c conda-forge ipython=7.19.0 --yes
conda install -c conda-forge jupyterlab=3.0.14 --yes
conda install -c conda-forge lightgbm=3.2.1 --yes
conda install -c conda-forge matplotlib=3.0.3 --yes
conda install -c conda-forge numpy=1.19.2 --yes
conda install -c conda-forge openpyxl=3.0.5 --yes
conda install -c conda-forge pandas=1.1.5 --yes
conda install -c conda-forge pillow=8.0.1 --yes
conda install -c conda-forge psutil --yes
conda install -c conda-forge python-language-server=0.36.2 --yes
conda install -c conda-forge pyyaml --yes
conda install -c conda-forge scikit-image=0.17.1 --yes
conda install -c conda-forge scikit-learn=0.24.0 --yes
conda install -c conda-forge seaborn=0.11.1 --yes
