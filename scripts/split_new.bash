#!/bin/bash

# All
# dataname=tidy_all

# Partially balanced
dataname=tidy_partially_balanced

datapath=data/processed/$dataname/annotations.csv
gout=data/processed/$dataname

split_on=slide
# split_on=Group

echo "datapath: $datapath"

# python /vol/ml/apartin/projects/pdx-histo/src/ml-data-splits/src/main_data_split.py \
#     --datapath $datapath \
#     --gout $gout \
#     --n_splits 5 \
#     --cv_method group \
#     --split_on $split_on \
#     --te_size 0.1 \
#     --ml_task cls \
#     --trg_name Response

python /vol/ml/apartin/projects/pdx-histo/src/ml-data-splits/src/main_data_split.py \
    --datapath $datapath \
    --gout $gout \
    --n_splits 5 \
    --cv_method simple \
    --te_size 0.1 \
    --ml_task cls \
    --trg_name Response

# python /vol/ml/apartin/projects/pdx-histo/src/ml-data-splits/src/main_data_split.py \
#     --datapath $datapath \
#     --gout $gout \
#     --n_splits 5 \
#     --cv_method strat \
#     --split_on $split_on \
#     --te_size 0.1 \
#     --ml_task cls \
#     --trg_name Response
