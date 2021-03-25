#!/bin/bash

# All
# dataname=tidy_all
# prjname=bin_rsp_all

# Partially balanced
dataname=tidy_partially_balanced
# prjname=bin_rsp_partially_balanced

datapath=data/processed/$dataname/annotations.csv
# prjdir=projects/$prjname
# gout=$prjdir
gout=data/processed/$dataname

split_on=slide

echo "datapath: $datapath"

# python ./src/ml-data-splits/src/main_data_split.py \
#     --datapath $datapath \
#     --gout $gout \
#     --n_splits 5 \
#     --cv_method strat \
#     --te_size 0.1 \
#     --ml_task cls \
#     --trg_name Response

python /vol/ml/apartin/projects/pdx-histo/src/ml-data-splits/src/main_data_split.py \
    --datapath $datapath \
    --gout $gout \
    --n_splits 5 \
    --cv_method group \
    --split_on $split_on \
    --te_size 0.1 \
    --ml_task cls \
    --trg_name Response
