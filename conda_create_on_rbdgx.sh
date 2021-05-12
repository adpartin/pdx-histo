#!/bin/bash

# Run script as follows:
# $ ./conda_create_on_lamina.sh

# Create env by running the line below, and then run this script.
# conda create -n pdx_rbdgx python=3.7 pip --yes

conda install -c anaconda tensorflow-gpu=2.4.1 --yes

conda install -c conda-forge ipdb=0.13.4 \
    ipython=7.19.0 \
    jupyterlab=3.0.14 \
    lightgbm=3.2.1 \
    matplotlib=3.0.3 \
    numpy=1.19.2 \
    openpyxl=3.0.5 \
    pandas=1.1.5 \
    pillow=8.0.1 \
    psutil=5.8 \
    python-language-server=0.36.2 \
    pyyaml=5.4.1 \
    scikit-image=0.17.1 \
    scikit-learn=0.24.0 \
    seaborn=0.11.1 --yes
