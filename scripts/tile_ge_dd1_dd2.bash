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

split_id=81
# split_id=0
# split_id=1
# split_id=2
# split_id=3

# Number of drug response samples
# n_samples=60
# n_samples=80
# n_samples=100
n_samples=-1

# Dir name that contains TFRecords
# tfr_dir_name=PDX_FIXED_RSP_DRUG_PAIR
tfr_dir_name=PDX_FIXED_RSP_DRUG_PAIR_0.1_of_tiles
# tfr_dir_name=PDX_FIXED_RSP_DRUG_PAIR_0.2_of_tiles
# tfr_dir_name=PDX_FIXED_RSP_DRUG_PAIR_0.4_of_tiles
# tfr_dir_name=PDX_FIXED_RSP_DRUG_PAIR_10_tiles

# Dir name that contains TFRecords for predictions
pred_tfr_dir_name=PDX_FIXED_RSP_DRUG_PAIR

DEVICE=$1
echo "CUDA device: $DEVICE"

trn_phase=$2
# nn_arch=$3

CUDA_VISIBLE_DEVICES=$DEVICE python src/trn_multimodal.py \
    --target $target \
    --split_on $split_on \
    --split_id $split_id \
    --id_name $id_name \
    --prjname $prjname \
    --trn_phase $trn_phase \
    --dataname $dataname \
    --n_samples $n_samples \
    --tfr_dir_name $tfr_dir_name \
    --pred_tfr_dir_name $pred_tfr_dir_name \
    --use_tile --use_ge --use_dd1 --use_dd2

    # --nn_arch $nn_arch \
