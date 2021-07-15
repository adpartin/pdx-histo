"""
Prediction of ctype with TFRecords.
"""
import os
import sys
assert sys.version_info >= (3, 5)

# https://www.codegrepper.com/code-examples/python/suppres+tensorflow+warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from collections import OrderedDict
import csv
import glob
from pathlib import Path
from pprint import pprint, pformat
import shutil
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
assert tf.__version__ >= "2.0"

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

fdir = Path(__file__).resolve().parent
# from config import cfg
sys.path.append(str(fdir/".."))
import src
from src.config import cfg
from src.models import (build_model_rsp, build_model_rsp_baseline, keras_callbacks, load_best_model,
                        calc_tf_preds, calc_smp_preds)
from src.models import calc_tile_preds, agg_tile_preds
from src.ml.scale import get_scaler
from src.ml.evals import calc_scores, save_confusion_matrix
from src.ml.keras_utils import plot_prfrm_metrics
from src.utils.classlogger import Logger
from src.utils.utils import (cast_list, create_outdir, create_outdir_2, dump_dict, fea_types_to_str_name,
                             get_print_func, read_lines, Params, Timer)
from src.datasets.tidy import split_data_and_extract_fea, extract_fea, TidyData
from src.tf_utils import get_tfr_files
from src.sf_utils import (create_manifest, create_tf_data, calc_class_weights,
                          parse_tfrec_fn_rsp, parse_tfrec_fn_rna, parse_tfrec_fn_ctype)
from src.sf_utils import bold, green, blue, yellow, cyan, red


# Seed
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)


def print_groupby_stat_rsp(df, split_on="Group", print_fn=print):
    print_fn(df.groupby(["ctype", "Response"]).agg({split_on: "nunique", "smp": "nunique"}).reset_index().rename(
        columns={split_on: f"{split_on}_unq", "smp": "smp_unq"}))


def print_groupby_stat_ctype(df, split_on="Group", print_fn=print):
    print_fn(df.groupby(["ctype", "ctype_label"]).agg({split_on: "nunique", "smp": "nunique", "slide": "nunique"}).reset_index().rename(
        columns={split_on: f"{split_on}_unq", "smp": "smp_unq", "slide": "slide_unq"}))


# Per-slide hits
def calc_hits(df, meta):
    df["hit"] = df.y_true == df.y_pred_label
    df = df.groupby("image_id").agg(hit_tiles=("hit", "sum"), n_tiles=("tile_id", "nunique")).reset_index()
    df["hit_rate"] = df["hit_tiles"] / df["n_tiles"]
    df = df.merge(meta[["image_id", "ctype_label", "ctype"]], on="image_id", how="inner")
    df = df.sort_values(["hit_rate"], ascending=False)
    return df


def parse_args(args):
    parser = argparse.ArgumentParser("Train NN.")
    parser.add_argument("-t", "--target",
                        type=str,
                        nargs='+',
                        default=["ctype_label"],
                        choices=["Response", "ctype_label", "csite"],
                        help="Name of target to be precited.")
    parser.add_argument("--id_name",
                        type=str,
                        default="smp",
                        choices=["slide", "smp"],
                        help="Column name of the ID.")
    parser.add_argument("--split_on",
                        type=str,
                        default="Group",
                        choices=["Sample", "slide", "Group"],
                        help="Specify the hard split variable/column (default: None).")
    # parser.add_argument("--split_id",
    #                     type=int,
    #                     default=81,
    #                     help="Split id of the unique split for T/V/E sets (default: 0).")
    parser.add_argument("--prjname",
                        type=str,
                        help="Project name (to the store the restuls.")
    parser.add_argument("--dataname",
                        type=str,
                        default="tidy_all",
                        help="Data name (folder that contains the annotations.csv dataframe).")
    parser.add_argument("--rundir",
                        type=str,
                        default=None,
                        help="Dir that contains the saved models (used for evaluation mode).")
    parser.add_argument("--train",
                        action="store_true",
                        help="Train mode.")
    parser.add_argument("--eval",
                        action="store_true",
                        help="Evaluate mode.")
    parser.add_argument("--n_samples",
                        type=int,
                        default=-1,
                        help="Total samples from tr_id to process.")
    parser.add_argument("--tfr_dir_name",
                        type=str,
                        default="PDX_FIXED_RSP_DRUG_PAIR_0.1_of_tiles",
                        help="Dir name that contains TFRecords that are used for training.")
    parser.add_argument("--pred_tfr_dir_name",
                        type=str,
                        default="PDX_FIXED_RSP_DRUG_PAIR",
                        help="Dir name that contains TFRecords that are used for prediction/evaluation.")
    parser.add_argument("--scale_fea",
                        action="store_true",
                        help="Scale features.")
    parser.add_argument("--use_tile",
                        action="store_true",
                        help="Use tiles.")
    parser.add_argument("--use_ge",
                        action="store_true",
                        help="Use gene expression.")
    parser.add_argument("--use_dd1",
                        action="store_true",
                        help="Use drug descriptors for drug 1.")
    parser.add_argument("--use_dd2",
                        action="store_true",
                        help="Use drug descriptors for drug 2.")
    parser.add_argument("--drop_single_drug",
                        action="store_true",
                        help="Whether to drop single drug treatments from training set.")
    parser.add_argument("--drop_drug_pair_aug",
                        action="store_true",
                        help="Whether to drop the augmented drug-pair samples from training set.")
    args, other_args = parser.parse_known_args(args)
    # args = parser.parse_args(args)
    return args


def run(args):
    split_on = "none" if args.split_on is (None or "none") else args.split_on


    # Create project dir (if it doesn't exist)
    # import ipdb; ipdb.set_trace()
    prjdir = cfg.MAIN_PRJDIR/args.prjname
    os.makedirs(prjdir, exist_ok=True)


    # Create outdir (using the loaded hyperparamters) or
    # use content (model) from an existing run
    fea_strs = ["use_tile"]
    args_dict = vars(args)
    fea_names = "_".join([k.split("use_")[-1] for k in fea_strs if args_dict[k] is True])
    prm_file_path = prjdir/f"params_{fea_names}.json"
    if prm_file_path.exists() is False:
        shutil.copy(fdir/f"../default_params/default_params_{fea_names}.json", prm_file_path)
    params = Params(prm_file_path)

    if args.rundir is not None:
        outdir = Path(args.rundir).resolve()
        assert outdir.exists(), f"The {outdir} doen't exist."
        print_fn = print
    else:
        # outdir = create_outdir(prjdir, args)
        outdir = prjdir/f"{params.base_image_model}_finetuned"

        # Save hyper-parameters
        params.save(outdir/"params.json")

        # Logger
        lg = Logger(outdir/"logger.log")
        print_fn = get_print_func(lg.logger)
        print_fn(f"File path: {fdir}")
        print_fn(f"\n{pformat(vars(args))}")


    # Load dataframe (annotations)
    annotations_file = cfg.DATA_PROCESSED_DIR/args.dataname/cfg.SF_ANNOTATIONS_FILENAME
    dtype = {"image_id": str, "slide": str}
    data = pd.read_csv(annotations_file, dtype=dtype, engine="c", na_values=["na", "NaN"], low_memory=True)
    # data = data.astype({"image_id": str, "slide": str})
    print_fn(data.shape)


    # print_fn("\nFull dataset:")
    # if args.target[0] == "Response":
    #     print_groupby_stat_rsp(data, split_on="Group", print_fn=print_fn)
    # else:
    #     print_groupby_stat_ctype(data, split_on="Group", print_fn=print_fn)
    print_groupby_stat_ctype(data, split_on="Group", print_fn=print_fn)


    # Drop slide dups
    fea_columns = ["slide"]
    data = data.drop_duplicates(subset=fea_columns)

    # Aggregate non-responders to balance the responders
    # import ipdb; ipdb.set_trace()
    # n_samples = data["ctype"].value_counts().min()
    n_samples = 30
    dfs = []
    for ctype, count in data['ctype'].value_counts().items():
        aa = data[data.ctype == ctype]
        if aa.shape[0] > n_samples:
            aa = aa.sample(n=n_samples)
        dfs.append(aa)
    data = pd.concat(dfs, axis=0).reset_index(drop=True)
    print_groupby_stat_ctype(data, split_on="Group", print_fn=print_fn)

    te_size = 0.15
    itr, ite = train_test_split(np.arange(data.shape[0]), test_size=te_size, shuffle=True, stratify=data["ctype_label"].values)
    tr_meta_ = data.iloc[itr,:].reset_index(drop=True)
    te_meta = data.iloc[ite,:].reset_index(drop=True)

    vl_size = 0.10
    itr, ivl = train_test_split(np.arange(tr_meta_.shape[0]), test_size=vl_size, shuffle=True, stratify=tr_meta_["ctype_label"].values)
    tr_meta = tr_meta_.iloc[itr,:].reset_index(drop=True)
    vl_meta = tr_meta_.iloc[ivl,:].reset_index(drop=True)

    print_groupby_stat_ctype(tr_meta, split_on="Group", print_fn=print_fn)
    print_groupby_stat_ctype(vl_meta, split_on="Group", print_fn=print_fn)
    print_groupby_stat_ctype(te_meta, split_on="Group", print_fn=print_fn)

    print_fn(tr_meta.shape)
    print_fn(vl_meta.shape)
    print_fn(te_meta.shape)

    # Determine tfr_dir (the path to TFRecords)
    tfr_dir = (cfg.DATADIR/args.tfr_dir_name).resolve()
    pred_tfr_dir = (cfg.DATADIR/args.pred_tfr_dir_name).resolve()
    label = f"{params.tile_px}px_{params.tile_um}um"
    tfr_dir = tfr_dir/label
    pred_tfr_dir = pred_tfr_dir/label

    # Scalers for each feature set
    ge_scaler, dd1_scaler, dd2_scaler = None, None, None

    ge_cols  = [c for c in data.columns if c.startswith("ge_")]
    dd1_cols = [c for c in data.columns if c.startswith("dd1_")]
    dd2_cols = [c for c in data.columns if c.startswith("dd2_")]

    if args.scale_fea:
        if args.use_ge and len(ge_cols) > 0:
            ge_scaler = get_scaler(data[ge_cols])
        if args.use_dd1 and len(dd1_cols) > 0:
            dd1_scaler = get_scaler(data[dd1_cols])
        if args.use_dd2 and len(dd2_cols) > 0:
            dd2_scaler = get_scaler(data[dd2_cols])


    # --------------------------
    # Obtain T/V/E tfr filenames
    # --------------------------
    # List of sample names for T/V/E
    tr_smp_names = list(tr_meta[args.id_name].values)
    vl_smp_names = list(vl_meta[args.id_name].values)
    te_smp_names = list(te_meta[args.id_name].values)

    # TFRecords filenames
    train_tfr_files = get_tfr_files(tfr_dir, tr_smp_names)
    val_tfr_files = get_tfr_files(tfr_dir, vl_smp_names)
    if args.eval is True:
        assert pred_tfr_dir.exists(), f"Dir {pred_tfr_dir} is not found."
        # test_tfr_files = get_tfr_files(tfr_dir, te_smp_names)  # use same tfr_dir for eval
        test_tfr_files = get_tfr_files(pred_tfr_dir, te_smp_names)
        # print_fn("Total samples {}".format(len(train_tfr_files) + len(val_tfr_files) + len(test_tfr_files)))

    assert sorted(tr_smp_names) == sorted(tr_meta[args.id_name].values.tolist()), "Sample names in the tr_smp_names and tr_meta don't match."
    assert sorted(vl_smp_names) == sorted(vl_meta[args.id_name].values.tolist()), "Sample names in the vl_smp_names and vl_meta don't match."
    assert sorted(te_smp_names) == sorted(te_meta[args.id_name].values.tolist()), "Sample names in the te_smp_names and te_meta don't match."


    # -------------------------------
    # Class weight
    # -------------------------------
    tile_cnts = pd.read_csv(tfr_dir/"tile_counts_per_slide.csv")
    tile_cnts.insert(loc=0, column="tfr_abs_fname", value=tile_cnts["tfr_fname"].map(lambda s: str(tfr_dir/s)))
    cat = tile_cnts[tile_cnts["tfr_abs_fname"].isin(train_tfr_files)]

    # import ipdb; ipdb.set_trace()
    ### ap --------------
    # if args.target[0] not in cat.columns:
    #     tile_cnts = tile_cnts[tile_cnts["smp"].isin(tr_meta["smp"])]
    df = tr_meta[["smp", args.target[0]]]
    cat = cat.merge(df, on="smp", how="inner")
    ### ap --------------

    cat = cat.groupby(args.target[0]).agg({"smp": "nunique", "max_tiles": "sum", "n_tiles": "sum", "slide": "nunique"}).reset_index()
    categories = {}
    for i, row_data in cat.iterrows():
        dct = {"num_samples": row_data["smp"], "num_tiles": row_data["n_tiles"]}
        categories[row_data[args.target[0]]] = dct

    class_weight = calc_class_weights(train_tfr_files,
                                      class_weights_method=params.class_weights_method,
                                      categories=categories)


    # --------------------------
    # Build tf.data objects
    # --------------------------
    tf.keras.backend.clear_session()

    # import ipdb; ipdb.set_trace()
    if args.use_tile:

        # -------------------------------
        # Parsing funcs
        # -------------------------------
        # import ipdb; ipdb.set_trace()
        if args.target[0] == "Response":
            # Response
            parse_fn = parse_tfrec_fn_rsp
            parse_fn_train_kwargs = {
                "use_tile": args.use_tile,
                "use_ge": args.use_ge,
                "use_dd1": args.use_dd1,
                "use_dd2": args.use_dd2,
                "ge_scaler": ge_scaler,
                "dd1_scaler": dd1_scaler,
                "dd2_scaler": dd2_scaler,
                "id_name": args.id_name,
                "augment": params.augment,
                "application": params.base_image_model,
                # "application": None,
            }
        else:
            # Ctype
            parse_fn = parse_tfrec_fn_ctype
            parse_fn_train_kwargs = {
                "use_tile": args.use_tile,
                "use_ge": args.use_ge,
                "ge_scaler": ge_scaler,
                "id_name": args.id_name,
                "augment": params.augment,
                "target": args.target[0]
            }

        parse_fn_non_train_kwargs = parse_fn_train_kwargs.copy()
        parse_fn_non_train_kwargs["augment"] = False

        # ----------------------------------------
        # Number of tiles/examples in each dataset
        # ----------------------------------------
        # import ipdb; ipdb.set_trace()
        tr_tiles = tile_cnts[tile_cnts[args.id_name].isin(tr_smp_names)]["n_tiles"].sum()
        vl_tiles = tile_cnts[tile_cnts[args.id_name].isin(vl_smp_names)]["n_tiles"].sum()
        te_tiles = tile_cnts[tile_cnts[args.id_name].isin(te_smp_names)]["n_tiles"].sum()

        eval_batch_size = 4 * params.batch_size
        tr_steps = tr_tiles // params.batch_size
        vl_steps = vl_tiles // eval_batch_size
        te_steps = te_tiles // eval_batch_size

        # -------------------------------
        # Create TF datasets
        # -------------------------------
        print("\nCreating TF datasets.")

        # Training
        # import ipdb; ipdb.set_trace()
        train_data = create_tf_data(
            batch_size=params.batch_size,
            deterministic=False,
            include_meta=False,
            interleave=True,
            n_concurrent_shards=params.n_concurrent_shards,  # 32, 64
            parse_fn=parse_fn,
            prefetch=1,  # 2
            repeat=True,
            seed=None,  # cfg.seed,
            shuffle_files=True,
            shuffle_size=params.shuffle_size,  # 8192
            tfrecords=train_tfr_files,
            **parse_fn_train_kwargs)

        # Determine feature shapes from data
        bb = next(train_data.__iter__())

        # Infer dims of features from the data
        # import ipdb; ipdb.set_trace()
        if args.use_ge:
            ge_shape = bb[0]["ge_data"].numpy().shape[1:]
        else:
            ge_shape = None

        if args.use_dd1:
            dd_shape = bb[0]["dd1_data"].numpy().shape[1:]
        else:
            dd_shape = None

        # Print keys and dims
        for i, item in enumerate(bb):
            print(f"\nItem {i}")
            if isinstance(item, dict):
                for k in item.keys():
                    print(f"\t{k}: {item[k].numpy().shape}")
            elif isinstance(item.numpy(), np.ndarray):
                print(item)

        # Evaluation (val, test, train)
        create_tf_data_eval_kwargs = {
            "batch_size": eval_batch_size,
            "include_meta": False,
            "interleave": False,
            "parse_fn": parse_fn,
            "prefetch": None,  # 2
            "repeat": False,
            "seed": None,
            "shuffle_files": False,
            "shuffle_size": None,
        }

        # import ipdb; ipdb.set_trace()
        create_tf_data_eval_kwargs.update({"tfrecords": val_tfr_files, "include_meta": False})
        val_data = create_tf_data(
            **create_tf_data_eval_kwargs,
            **parse_fn_non_train_kwargs
        )

    # ----------------------
    # Prep for training
    # ----------------------
    # import ipdb; ipdb.set_trace()

    # -------------
    # Train model
    # -------------
    model = None

    # import ipdb; ipdb.set_trace()
    if args.train is True:

        # Callbacks list
        monitor = "val_loss"
        callbacks = keras_callbacks(outdir, monitor=monitor,
                                    save_best_only=params.save_best_only,
                                    patience=params.patience)

        # Mixed precision
        if params.use_fp16:
            print_fn("\nTrain with mixed precision")
            if int(tf.keras.__version__.split(".")[1]) == 4:  # TF 2.4
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy("mixed_float16")
                mixed_precision.set_global_policy(policy)
            elif int(tf.keras.__version__.split(".")[1]) == 3:  # TF 2.3
                from tensorflow.keras.mixed_precision import experimental as mixed_precision
                policy = mixed_precision.Policy("mixed_float16")
                mixed_precision.set_policy(policy)
            print_fn("Compute dtype: %s" % policy.compute_dtype)
            print_fn("Variable dtype: %s" % policy.variable_dtype)

        # ----------------------
        # Define model
        # ----------------------
        # import ipdb; ipdb.set_trace()

        from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
        from tensorflow.keras import layers
        from tensorflow.keras import losses
        from tensorflow.keras import optimizers
        from tensorflow.keras.models import Sequential, Model, load_model

        # trainable = True
        trainable = False
        # from_logits = True
        from_logits = False
        fit_verbose = 1
        pretrain = params.pretrain
        pooling = params.pooling
        n_classes = len(sorted(tr_meta[args.target[0]].unique()))

        model_inputs = []
        merge_inputs = []

        if args.use_tile:
            image_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
            tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")

            base_img_model = tf.keras.applications.Xception(
                include_top=False,
                weights=pretrain,
                input_shape=None,
                input_tensor=None,
                pooling=pooling)

            print_fn(f"\nNumber of layers in the base image model ({params.base_image_model}): {len(base_img_model.layers)}")
            print_fn("Trainable variables: {}".format(len(base_img_model.trainable_variables)))
            print_fn("Shape of trainable variables at {}: {}".format(0, base_img_model.trainable_variables[0].shape))
            print_fn("Shape of trainable variables at {}: {}".format(-1, base_img_model.trainable_variables[-1].shape))

            print_fn("\nFreeze base model.")
            base_img_model.trainable = trainable  # Freeze the base_img_model
            print_fn("Trainable variables: {}".format(len(base_img_model.trainable_variables)))

            print_fn("\nPrint some layers")
            print_fn("Name of layer {}: {}".format(0, base_img_model.layers[0].name))
            print_fn("Name of layer {}: {}".format(-1, base_img_model.layers[-1].name))

            # training=False makes the base model to run in inference mode so
            # that batchnorm layers are not updated during the fine-tuning stage.
            # x_tile = base_img_model(tile_input_tensor)
            x_tile = base_img_model(tile_input_tensor, training=False)
            # x_tile = base_img_model(tile_input_tensor, training=trainable)
            model_inputs.append(tile_input_tensor)

            # x_tile = Dense(params.dense1_img, activation=tf.nn.relu, name="dense1_img")(x_tile)
            # x_tile = Dense(params.dense2_img, activation=tf.nn.relu, name="dense2_img")(x_tile)
            # x_tile = BatchNormalization(name="batchnorm_im")(x_tile)
            merge_inputs.append(x_tile)
            del tile_input_tensor, x_tile

        # Merge towers
        if len(merge_inputs) > 1:
            mm = layers.Concatenate(axis=1, name="merger")(merge_inputs)
        else:
            mm = merge_inputs[0]

        # Dense layers of the top classfier
        mm = Dense(params.dense1_top, activation=tf.nn.relu, name="dense1_top")(mm)
        # mm = BatchNormalization(name="batchnorm_top")(mm)
        # mm = Dropout(params.dropout1_top)(mm)

        # Output
        output = Dense(n_classes, activation=tf.nn.relu, name="logits")(mm)
        if from_logits is False:
            output = Activation(tf.nn.softmax, dtype="float32", name="softmax")(output)

        # Assemble final model
        model = Model(inputs=model_inputs, outputs=output)

        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="CatAcc"),
            tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits, name="CatCrossEnt")
        ]

        if params.optimizer == "SGD":
            optimizer = optimizers.SGD(learning_rate=params.learning_rate, momentum=0.9, nesterov=True)
        elif params.optimizer == "Adam":
            optimizer = optimizers.Adam(learning_rate=params.learning_rate)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


        # import ipdb; ipdb.set_trace()
        print_fn("\nBase model")
        base_img_model.summary(print_fn=print_fn)
        print_fn("\nFull model")
        model.summary(print_fn=print_fn)
        print_fn("Trainable variables: {}".format(len(model.trainable_variables)))

        print_fn(f"Train steps:      {tr_steps}")
        # print_fn(f"Validation steps: {vl_steps}")

        # ------------
        # Train
        # ------------
        import ipdb; ipdb.set_trace()
        # tr_steps = 10  # tr_tiles // params.batch_size // 15  # for debugging
        print_fn("\n{}".format(yellow("Train")))
        timer = Timer()
        history = model.fit(x=train_data,
                            validation_data=val_data,
                            steps_per_epoch=tr_steps,
                            validation_steps=vl_steps,
                            class_weight=class_weight,
                            epochs=params.epochs,
                            verbose=fit_verbose,
                            callbacks=callbacks)
        # del train_data, val_data
        timer.display_timer(print_fn)
        plot_prfrm_metrics(history, title="Train stage", name="tn", outdir=outdir)
        model = load_best_model(outdir)  # load best model

        # Save trained model
        print_fn("\nSave trained model.")
        model.save(outdir/"best_model_trained")

        create_tf_data_eval_kwargs.update({"tfrecords": test_tfr_files, "include_meta": True})
        test_data = create_tf_data(
            **create_tf_data_eval_kwargs,
            **parse_fn_non_train_kwargs
        )

        # Calc hits
        te_tile_preds = calc_tile_preds(test_data, model=model, outdir=outdir)
        te_tile_preds = te_tile_preds.sort_values(["image_id", "tile_id"], ascending=True)
        hits_tn = calc_hits(te_tile_preds, te_meta)
        hits_tn.to_csv(outdir/"hits_tn.csv", index=False)

        # ------------
        # Finetune
        # ------------
        # import ipdb; ipdb.set_trace()
        print_fn("\n{}".format(green("Finetune")))
        unfreeze_top_layers = 50
        # Unfreeze layers of the base model
        for layer in base_img_model.layers[-unfreeze_top_layers:]:
            layer.trainable = True
            print_fn("{}: (trainable={})".format(layer.name, layer.trainable))
        print_fn("Trainable variables: {}".format(len(model.trainable_variables)))

        model.compile(loss=loss,
                      optimizer=optimizers.Adam(learning_rate=params.learning_rate/10),
                      metrics=metrics)

        callbacks = keras_callbacks(outdir, monitor=monitor,
                                    save_best_only=params.save_best_only,
                                    patience=params.patience,
                                    name="finetune")

        total_epochs = history.epoch[-1] + params.finetune_epochs
        timer = Timer()
        history_fn = model.fit(x=train_data,
                               validation_data=val_data,
                               steps_per_epoch=tr_steps,
                               validation_steps=vl_steps,
                               class_weight=class_weight,
                               epochs=total_epochs,
                               initial_epoch=history.epoch[-1]+1,
                               verbose=fit_verbose,
                               callbacks=callbacks)
        del train_data, val_data
        plot_prfrm_metrics(history_fn, title="Finetune stage", name="fn", outdir=outdir)
        timer.display_timer(print_fn)

        # Save trained model
        print_fn("\nSave finetuned model.")
        model.save(outdir/"best_model_finetuned")
        base_img_model.save(outdir/"best_model_img_base_finetuned")


    if args.eval is True:

        print_fn("\n{}".format(bold("Test set predictions.")))
        timer = Timer()
        # import ipdb; ipdb.set_trace()
        te_tile_preds = calc_tile_preds(test_data, model=model, outdir=outdir)
        te_tile_preds = te_tile_preds.sort_values(["image_id", "tile_id"], ascending=True)
        te_tile_preds.to_csv(outdir/"te_tile_preds.csv", index=False)
        # print(te_tile_preds[["image_id", "tile_id", "y_true", "y_pred_label", "prob"]][:20])
        # print(te_tile_preds.iloc[:20, 1:])
        del test_data

        # Calc hits
        hits_fn = calc_hits(te_tile_preds, te_meta)
        hits_fn.to_csv(outdir/"hits_fn.csv", index=False)

        # import ipdb; ipdb.set_trace()
        roc_auc = {}
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        fig, ax = plt.subplots(figsize=(8, 6))
        for true in range(0, n_classes):
            if true in te_tile_preds["y_true"].values:
                fpr, tpr, thresh = roc_curve(te_tile_preds["y_true"], te_tile_preds["prob"], pos_label=true)
                roc_auc[i] = auc(fpr, tpr)
                plt.plot(fpr, tpr, linestyle='--', label=f"Class {true} vs Rest")
            else:
                roc_auc[i] = None

        # plt.plot([0,0], [1,1], '--', label="Random")
        plt.title("Multiclass ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc="best")
        plt.savefig(outdir/"Multiclass ROC", dpi=70);

        # Avergae precision score
        from sklearn.metrics import average_precision_score
        y_true_vec = te_tile_preds.y_true.values
        y_true_onehot = np.zeros((y_true_vec.size, n_classes))
        y_true_onehot[np.arange(y_true_vec.size), y_true_vec] = 1
        y_probs = te_tile_preds[[c for c in te_tile_preds.columns if "prob_" in c]]
        print_fn("\nAvearge precision")
        print_fn("Micro    {}".format(average_precision_score(y_true_onehot, y_probs, average="micro")))
        print_fn("Macro    {}".format(average_precision_score(y_true_onehot, y_probs, average="macro")))
        print_fn("Wieghted {}".format(average_precision_score(y_true_onehot, y_probs, average="weighted")))
        print_fn("Samples  {}".format(average_precision_score(y_true_onehot, y_probs, average="samples")))


        import ipdb; ipdb.set_trace()
        agg_method = "mean"
        # agg_by = "smp"
        agg_by = "image_id"
        smp_preds = agg_tile_preds(te_tile_preds, agg_by=agg_by, meta=te_meta, agg_method=agg_method)

        timer.display_timer(print_fn)

    lg.close_logger()


def main(args):
    args = parse_args(args)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
