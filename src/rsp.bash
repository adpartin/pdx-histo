#!/bin/bash

split_on=slide
target=Response
id_name=smp

DEVICE=$1
echo "CUDA device: $DEVICE"
CUDA_VISIBLE_DEVICES=$DEVICE python /vol/ml/apartin/projects/pdx-histo/src/trn_rsp.py \
    --target $target \
    --id_name $id_name \
    --split_on $split_on
