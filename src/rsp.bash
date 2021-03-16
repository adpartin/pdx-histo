#!/bin/bash

DEVICE=$1
echo "CUDA device: $DEVICE"
CUDA_VISIBLE_DEVICES=$DEVICE python /vol/ml/apartin/projects/pdx-histo/src/trn_rsp.py \
    --target Response --id_name smp
