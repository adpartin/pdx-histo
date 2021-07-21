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

# split_id=81
# split_id=0
# split_id=3

# Number of drug response samples
# n_samples=60
# n_samples=80
# n_samples=100
n_samples=-1

DEVICE=$1
echo "CUDA device: $DEVICE"

# CUDA_VISIBLE_DEVICES=$DEVICE python src/trn_multimodal.py \
#     --train \
#     --eval \
#     --target $target \
#     --split_on $split_on \
#     --split_id $split_id \
#     --id_name $id_name \
#     --prjname $prjname \
#     --dataname $dataname \
#     --n_samples $n_samples \
#     --use_ge --use_dd1 --use_dd2

# split_id=81
# split_id=0
split_id=$2

# -----------
# Train
# -----------
# n_splits=4
# n_splits=99
# splits_arr=($(seq 0 1 $n_splits))
splits_arr=($split_id)
# echo -e "\nList of splits:"
# echo -e "${splits_arr[@]}\n"

# for ii in {0..${n_splits}}; do
# for ii in {0..99}; do
for split_id in ${splits_arr[@]}; do
    echo -e "Split ${split_id}"

    CUDA_VISIBLE_DEVICES=$DEVICE python src/trn_multimodal.py \
        --train \
        --eval \
        --target $target \
        --split_on $split_on \
        --split_id $split_id \
        --id_name $id_name \
        --prjname $prjname \
        --dataname $dataname \
        --n_samples $n_samples \
        --use_ge --use_dd1 --use_dd2

        # --tfr_dir_name $tfr_dir_name \
        # --pred_tfr_dir_name $pred_tfr_dir_name \

done
