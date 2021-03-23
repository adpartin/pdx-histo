#!/bin/bash

DEVICE=$1
echo "CUDA device: $DEVICE"
CUDA_VISIBLE_DEVICES=$DEVICE python /vol/ml/apartin/projects/pdx-histo/src/trn_rsp_baseline.py \
    --target Response \
    --split_on slide \
    --id_name smp
