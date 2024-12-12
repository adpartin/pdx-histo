Multimodal neural network for drug response prediction in PDX with histology images and gene expressions.

## Build tidy dataframe of drug response samples for binary classification
The dataframe will be stored in `data/processed`.
```
$ python scripts/build_tidy_drug_pairs_all_samples.py
```

## Generate TFRecords for drug response from TFRecords of WSIs
The TFRecords will be stored in `data/PDX_FIXED_RSP_DRUG_PAIR*`.
Depending on the input args, the designated folder will be created.
For example, the following command will create `data/PDX_FIXED_RSP_DRUG_PAIR_rnd_glob_min`.
```
$ python scripts/update_tfrecords.py --random --take_glob_min
```

## Specify Hyperparameters (HPs)
You can specify the HPs in ...


## Train baselines (analysis of data augmentation)
```
$ ./ge_dd1_dd2_drop_aug.bash
```
```
$ ./ge_dd1_dd2_only_pairs.bash
```

## Train multimodal deep learning model
Specify the parameters in the bash script as necessary.
```
./scripts/tile_ge_dd1_dd2.bash
```
The results will be bumped into `projects/bin_rsp_drug_pairs_all_samples`.
