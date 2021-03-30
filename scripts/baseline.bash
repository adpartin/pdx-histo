#!/bin/bash

# Data name
# dataname=tidy_all
dataname=tidy_partially_balanced

# Project name
# prjname=bin_rsp_all
prjname=bin_rsp_partially_balanced

id_name=smp
target=Response

# split_on=slide
split_on=Group

DEVICE=$1
echo "CUDA device: $DEVICE"
CUDA_VISIBLE_DEVICES=$DEVICE python /vol/ml/apartin/projects/pdx-histo/src/trn_baseline.py \
    --target $target \
    --split_on $split_on \
    --id_name $id_name \
    --prjname $prjname \
    --dataname $dataname
