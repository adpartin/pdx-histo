#!/bin/bash
# TODO: didn't finish
# stackoverflow.com/questions/885620/in-bash-how-can-i-print-the-first-n-elements-of-a-list

prjdir=projects/bin_rsp_drug_pairs_all_samples

# splits_dir_list=`ls $prjdir | grep split_ --include="tile_ge_dd1_dd2"`  # creates string
splits_dir_list=($(ls $prjdir | grep split_ --include="tile_ge_dd1_dd2"))  # creates array
echo -e "\nList of splits:"
echo -e "${splits_dir_list}\n"

total_splits=`ls $prjdir | grep split_ --include="tile_ge_dd1_dd2" | wc -l`
echo -e "\nTotal models: ${total_splits}\n"

# How many models to run inference for
start_split=0
n_splits=2
echo -e "start_split: $start_split"
echo -e "n_splits: $n_splits\n"

# for split_dir in ${splits_dir_list[@]}; do
# # for split_dir in $(seq $start_split 1 {splits_dir_list[@]}; do

# ----------------
# Python paramters
# ----------------
dataname=tidy_drug_pairs_all_samples
prjname=bin_rsp_drug_pairs_all_samples
id_name=smp
target=Response
split_on=Group
n_samples=-1
tfr_dir_name=PDX_FIXED_RSP_DRUG_PAIR_0.1_of_tiles
pred_tfr_dir_name=PDX_FIXED_RSP_DRUG_PAIR
split_id=$2  # TODO: problem!
# ----------------------------------------------------------------

for split_dir in ${splits_dir_list[@]:0:$n_splits}; do
    model_dir=$prjdir/$split_dir/final_model.ckpt
    echo -e "${model_dir}"

    if [[ -d "${model_dir}" ]]; then
        echo -e "\tInference."
        # CUDA_VISIBLE_DEVICES=$DEVICE python src/trn_multimodal.py \
        #     --eval \
        #     --rundir model_dir \
        #     --target $target \
        #     --split_on $split_on \
        #     --split_id $split_id \
        #     --id_name $id_name \
        #     --prjname $prjname \
        #     --dataname $dataname \
        #     --n_samples $n_samples \
        #     --tfr_dir_name $tfr_dir_name \
        #     --pred_tfr_dir_name $pred_tfr_dir_name \
        #     --use_tile --use_ge --use_dd1 --use_dd2
    else
        echo -e "\tModel not found."
    fi
done

# for ii in ${n_splits}; do
#     model_dir=$prjdir/${splits_dir_list[ii]}/final_model.ckpt
#     echo -e "${model_dir}"
#     if [[ -d "${model_dir}" ]]; then
#         echo -e "\tInference."
#     else
#         echo -e "\tModel not found."
#     fi
# done
