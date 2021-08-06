#!/bin/bash
# TODO: didn't finish
# stackoverflow.com/questions/885620/in-bash-how-can-i-print-the-first-n-elements-of-a-list

DEVICE=$1
echo "CUDA device: $DEVICE"

prjdir=projects/bin_rsp_drug_pairs_all_samples
prjdir=${prjdir}/runs_tile_dd
# include="tile_dd1_dd2"

# splits_dir_list=`ls $prjdir | grep split_ --include="tile_ge_dd1_dd2"`  # creates string
splits_dir_list=`ls $prjdir | grep split_ --include="tile_dd1_dd2"`  # creates string
# splits_dir_list=($(ls $prjdir | grep split_ --include=$include))  # creates array

echo -e "\nList of split dirs:"
echo -e "${splits_dir_list}\n"

total_splits=`ls $prjdir | grep split_ --include="tile_dd1_dd2" | wc -l`
echo -e "\nTotal models: ${total_splits}\n"

# How many models to run inference for

# split_start=0
# split_end=49

split_start=50
split_end=99

splits_arr=($(seq $split_start 1 $split_end))
# n_splits=2
echo -e "split_start: $split_start"
echo -e "split_end:   $split_end\n"
# echo -e "n_splits: $n_splits\n"

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
# split_id=$2  # TODO: problem!
# ----------------------------------------------------------------

for split_id in ${splits_arr[@]}; do
    echo -e "Split ${split_id}"
    # split_dir=`ls $prjdir | grep split_${split_id} --include="tile_dd1_dd2"`
    split_dir=`ls $prjdir | grep split_${split_id}_tile_dd1_dd2`
    echo -e "split_dir: $split_dir"
    model_dir=$prjdir/$split_dir/final_model.ckpt
    echo -e "model_dir: $model_dir"
    rundir=$prjdir/$split_dir

    if [[ -d "${model_dir}" ]]; then
        echo -e "\tInference."

        CUDA_VISIBLE_DEVICES=$DEVICE python src/trn_multimodal.py \
            --eval \
            --rundir $rundir \
            --target $target \
            --split_on $split_on \
            --split_id $split_id \
            --id_name $id_name \
            --prjname $prjname \
            --dataname $dataname \
            --n_samples $n_samples \
            --tfr_dir_name $tfr_dir_name \
            --pred_tfr_dir_name $pred_tfr_dir_name \
            --use_tile --use_dd1 --use_dd2

    else
        echo -e "\tModel not found."
    fi

done

# for split_dir in ${splits_dir_list[@]:0:$n_splits}; do
#     model_dir=$prjdir/$split_dir/final_model.ckpt
#     echo -e "${model_dir}"

#     if [[ -d "${model_dir}" ]]; then
#         echo -e "\tInference."
#         # CUDA_VISIBLE_DEVICES=$DEVICE python src/trn_multimodal.py \
#         #     --eval \
#         #     --rundir model_dir \
#         #     --target $target \
#         #     --split_on $split_on \
#         #     --split_id $split_id \
#         #     --id_name $id_name \
#         #     --prjname $prjname \
#         #     --dataname $dataname \
#         #     --n_samples $n_samples \
#         #     --tfr_dir_name $tfr_dir_name \
#         #     --pred_tfr_dir_name $pred_tfr_dir_name \
#         #     --use_tile --use_ge --use_dd1 --use_dd2
#     else
#         echo -e "\tModel not found."
#     fi
# done

# for ii in ${n_splits}; do
#     model_dir=$prjdir/${splits_dir_list[ii]}/final_model.ckpt
#     echo -e "${model_dir}"
#     if [[ -d "${model_dir}" ]]; then
#         echo -e "\tInference."
#     else
#         echo -e "\tModel not found."
#     fi
# done
