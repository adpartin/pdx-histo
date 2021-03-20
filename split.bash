#!/bin/bash

# appname=bin_rsp_balance_01
appname=bin_rsp_balance_02

appdir=apps/$appname
datapath=$appdir/annotations.csv
gout=$appdir

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

python ./src/ml-data-splits/src/main_data_split.py \
    --datapath $datapath \
    --gout $gout \
    --n_splits 5 \
    --cv_method group \
    --split_on $split_on \
    --te_size 0.1 \
    --ml_task cls \
    --trg_name Response
