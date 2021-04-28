#!/bin/bash

# Data name
# dataname=tidy_all
# dataname=tidy_partially_balanced
# dataname=tidy_single_drug_all_samples
dataname=tidy_drug_pairs_all_samples

# Project name
# prjname=bin_rsp_all
# prjname=bin_rsp_partially_balanced
# prjname=bin_rsp_single_drug_all_samples
prjname=bin_rsp_drug_pairs_all_samples

id_name=smp
target=Response

# split_on=slide
split_on=Group

# n_samples=60
# n_samples=80
# n_samples=100
n_samples=-1

DEVICE=$1
echo "CUDA device: $DEVICE"

trn_phase=$2
nn_arch=$3

# CUDA_VISIBLE_DEVICES=$DEVICE python /vol/ml/apartin/projects/pdx-histo/src/trn_multimodal2.py \
CUDA_VISIBLE_DEVICES=$DEVICE python src/trn_multimodal2.py \
    --target $target \
    --split_on $split_on \
    --id_name $id_name \
    --prjname $prjname \
    --trn_phase $trn_phase \
    --dataname $dataname \
    --nn_arch $nn_arch \
    --n_samples $n_samples \
    --use_tile --use_ge
