from pathlib import Path
from typing import Optional, List

import tensorflow as tf
assert tf.__version__ >= "2.0"
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# AUTO = tf.data.experimental.AUTOTUNE

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

fdir = Path(__file__).resolve().parent
# from config import cfg
from src.config import cfg
from src.sf_utils import bold, green, blue, yellow, cyan, red
from src.ml.evals import calc_scores, save_confusion_matrix
from src.utils.utils import  dump_dict

ModelDict = {
    "Xception": tf.keras.applications.Xception,
    "ResNet50": tf.keras.applications.ResNet50,
    "ResNet50V2": tf.keras.applications.ResNet50V2,
    "ResNet101": tf.keras.applications.ResNet101,
    "ResNet101V2": tf.keras.applications.ResNet101V2,
    "EfficientNetB1": tf.keras.applications.EfficientNetB1,
    "EfficientNetB2": tf.keras.applications.EfficientNetB2,
    "EfficientNetB3": tf.keras.applications.EfficientNetB3,
    "EfficientNetB4": tf.keras.applications.EfficientNetB4
}


def keras_callbacks(outdir, monitor="val_loss", save_best_only=True, patience=5, fname=None):
    """ ... """
    callbacks = []

    csv_logger = CSVLogger(outdir/"training.log")
    callbacks.append(csv_logger)

    if monitor == "val_pr-auc":
        mode = "max"
    elif monitor == "val_loss":
        mode = "min"
    else:
        mode = "auto"

    # filepath = str(outdir/"model_{epoch:02d}-{val_loss:.3f}.ckpt")
    if save_best_only is True:
        if fname is None:
            filepath = str(outdir/f"best_model.ckpt")
        else:
            filepath = str(outdir/fname)
    else:
        filepath = str(outdir/"model_{epoch:02d}-{val_loss:.3f}.ckpt")
    checkpointer = ModelCheckpoint(filepath,
                                   monitor=monitor,
                                   verbose=0,
                                   mode=mode,
                                   save_weights_only=False,
                                   save_best_only=save_best_only,
                                   save_freq="epoch")
    callbacks.append(checkpointer)

    reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                  factor=0.5,
                                  patience=5,
                                  verbose=1,
                                  mode=mode,
                                  min_delta=0.0001,
                                  cooldown=0,
                                  min_lr=0)
    callbacks.append(reduce_lr)

    early_stop = EarlyStopping(monitor=monitor,
                               patience=patience,
                               mode=mode,
                               restore_best_weights=True,
                               verbose=1)
    callbacks.append(early_stop)

    return callbacks


def load_best_model(models_dir, ckpt_name="best_model.ckpt", verbose=True, print_fn=print):
    """ Load the best checkpointed model where best is defined as a model with
    the lowest val_loss. The names of checkpointed models follow the same naming
    convention that contains the val_loss: model_{epoch:02d}-{val_loss:.3f}.ckpt
    """
    if (models_dir/ckpt_name).exists():
        model_path = models_dir/ckpt_name
        model = tf.keras.models.load_model(model_path)
    else:
        model_paths = sorted(models_dir.glob("model*.ckpt"))
        values = np.array([float(p.name.split(".ckpt")[0].split("-")[1]) for p in model_paths])
        # best_value = min(values)
        model_path = model_paths[np.argmin(values)]
        model = tf.keras.models.load_model(model_path)
    if verbose:
        print_fn(f"Loading model from: {model_path}")
    return model


def build_model_rsp_baseline(use_ge=True, use_dd1=True, use_dd2=True,
                             ge_shape=None, dd_shape=None, model_type="categorical",
                             dense1_ge=512, dense1_dd1=256, dense1_dd2=256, dense1_top=500,
                             dropout1_top=0.1,
                             # NUM_CLASSES=None,
                             output_bias=None):
    """ Doesn't use image data. """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model_inputs = []
    merge_inputs = []

    if use_ge:
        ge_input_tensor = tf.keras.Input(shape=ge_shape, name="ge_data")
        x_ge = Dense(dense1_ge, activation=tf.nn.relu, name="dense1_ge")(ge_input_tensor)
        x_ge = BatchNormalization(name="ge_batchnorm")(x_ge)
        # x_ge = Dropout(0.4)(x_ge)
        model_inputs.append(ge_input_tensor)
        merge_inputs.append(x_ge)
        del ge_input_tensor, x_ge

    if use_dd1:
        dd1_input_tensor = tf.keras.Input(shape=dd_shape, name="dd1_data")
        x_dd1 = Dense(dense1_dd1, activation=tf.nn.relu, name="dense1_dd1")(dd1_input_tensor)
        x_dd1 = BatchNormalization(name="dd1_batchnorm")(x_dd1)
        # x_dd1 = Dropout(0.4)(x_dd1)
        model_inputs.append(dd1_input_tensor)
        merge_inputs.append(x_dd1)
        del dd1_input_tensor, x_dd1

    if use_dd2:
        dd2_input_tensor = tf.keras.Input(shape=dd_shape, name="dd2_data")
        x_dd2 = Dense(dense1_dd2, activation=tf.nn.relu, name="dense1_dd2")(dd2_input_tensor)
        x_dd2 = BatchNormalization(name="dd2_batchnorm")(x_dd2)
        # x_dd2 = Dropout(0.4)(x_dd2)
        model_inputs.append(dd2_input_tensor)
        merge_inputs.append(x_dd2)
        del dd2_input_tensor, x_dd2

    # Merge towers
    merged_model = layers.Concatenate(axis=1, name="merger")(merge_inputs)

    merged_model = tf.keras.layers.Dense(dense1_top, activation=tf.nn.relu,
                                         name="dense1_top", kernel_regularizer=None)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    if dropout1_top > 0:
        merged_model = Dropout(dropout1_top)(merged_model)

    # Add the softmax prediction layer
    # activation = "linear" if model_type == "linear" else "softmax"
    # final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
    # softmax_output = tf.keras.layers.Activation(activation, dtype="float32", name="Response")(final_dense_layer)

    softmax_output = tf.keras.layers.Dense(
        1, activation="sigmoid", bias_initializer=output_bias, name="Response")(merged_model)

    # Assemble final model
    model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)
    return model


def build_model_rsp(use_ge=True, use_dd1=True, use_dd2=True, use_tile=True,
                    ge_shape=None, dd_shape=None,
                    dense1_img=1024, dense2_img=512,
                    dense1_ge=500,
                    dense1_dd1=250, dense1_dd2=250,
                    dense1_top=1000,
                    dropout1_top=0.1,
                    output_bias=None,
                    # model_type="categorical",
                    base_image_model="Xception",
                    pooling="max",
                    pretrain="imagenet",
                    loss=losses.BinaryCrossentropy(),
                    optimizer="SGD",
                    learning_rate=0.0001,
                    from_logits=False):
    """ ...
    refs:
        https://github.com/jkjung-avt/keras_imagenet/blob/master/utils/dataset.py
    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        
    model_inputs = []
    merge_inputs = []

    if use_tile:
        image_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
        tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")

        if pretrain == "imagenet":
            base_img_model = ModelDict[base_image_model](
                include_top=False,
                weights=pretrain,
                input_shape=None,
                input_tensor=None,
                pooling=pooling)
        else:
            base_img_model = ModelDict[base_image_model](
                include_top=False,
                weights=None,
                input_shape=None,
                input_tensor=None,
                pooling=pooling)
            base_img_model.load_weights(pretrain)

        base_img_model.trainable = False  # Freeze the base_img_model

        # training=False makes the base model to run in inference mode so
        # that batchnorm layers are not updated during the fine-tuning stage.
        # x_tile = base_img_model(tile_input_tensor)
        x_tile = base_img_model(tile_input_tensor, training=False)
        model_inputs.append(tile_input_tensor)

        if dense1_img > 0:
            x_tile = Dense(dense1_img, activation=tf.nn.relu, name="dense1_img")(x_tile)
        if dense2_img > 0:
            x_tile = Dense(dense2_img, activation=tf.nn.relu, name="dense2_img")(x_tile)
        if (dense1_img > 0) or (dense2_img > 0):
            x_tile = BatchNormalization(name="batchnorm_im")(x_tile)
        merge_inputs.append(x_tile)
        del tile_input_tensor, x_tile

    if use_ge:
        ge_input_tensor = tf.keras.Input(shape=ge_shape, name="ge_data")
        x_ge = Dense(dense1_ge, activation=tf.nn.relu, name="dense1_ge")(ge_input_tensor)
        x_ge = BatchNormalization(name="batchnorm_ge")(x_ge)
        model_inputs.append(ge_input_tensor)
        merge_inputs.append(x_ge)
        del ge_input_tensor, x_ge

    if use_dd1:
        dd1_input_tensor = tf.keras.Input(shape=dd_shape, name="dd1_data")
        x_dd1 = Dense(dense1_dd1, activation=tf.nn.relu, name="dense1_dd1")(dd1_input_tensor)
        x_dd1 = BatchNormalization(name="batchnorm_dd1")(x_dd1)
        model_inputs.append(dd1_input_tensor)
        merge_inputs.append(x_dd1)
        del dd1_input_tensor, x_dd1

    if use_dd2:
        dd2_input_tensor = tf.keras.Input(shape=dd_shape, name="dd2_data")
        x_dd2 = Dense(dense1_dd2, activation=tf.nn.relu, name="dense1_dd2")(dd2_input_tensor)
        x_dd2 = BatchNormalization(name="batchnorm_dd2")(x_dd2)
        model_inputs.append(dd2_input_tensor)
        merge_inputs.append(x_dd2)
        del dd2_input_tensor, x_dd2

    # Merge towers
    merged_model = layers.Concatenate(axis=1, name="merger")(merge_inputs)

    # Dense layers of the top classfier
    merged_model = Dense(dense1_top, activation=tf.nn.relu, name="dense1_top")(merged_model)
    merged_model = BatchNormalization(name="batchnorm_top")(merged_model)
    if dropout1_top > 0:
        merged_model = Dropout(dropout1_top)(merged_model)

    # Output
    # output = tf.keras.layers.Dense(
    #     1, activation="sigmoid", bias_initializer=output_bias, name="Response")(merged_model)

    output = Dense(1, name="logits")(merged_model)
    if from_logits:
        output = Activation(tf.nn.sigmoid, name="Response")(output)

    # Assemble final model
    model = Model(inputs=model_inputs, outputs=output)

    metrics = [
          # keras.metrics.FalsePositives(name="fp"),
          # keras.metrics.TruePositives(name="tp"),
          keras.metrics.AUC(name="roc-auc", curve="ROC"),
          keras.metrics.AUC(name="prc-auc", curve="PR"),
    ]

    if optimizer == "SGD":
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "Adam":
        optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def calc_tile_preds(tf_data_with_meta, model, outdir, p=0.5, verbose=True):
    """ ... """
    # meta_keys = ["smp", "Group", "grp_name", "Response"]
    # meta_keys = ["smp", "Group", "grp_name", "image_id", "tile_id"]
    # meta_keys = ["smp", "tile_id"]
    meta_keys = ["smp", "image_id", "tile_id"]
    # meta_keys = ["smp", "Group", "image_id", "tile_id"]
    meta_agg = {k: None for k in meta_keys}
    y_true, y_pred_prob, y_pred_label = [], [], []

    # import ipdb; ipdb.set_trace()
    for i, batch in enumerate(tf_data_with_meta):
        if (i+1) % 50 == 0:
            print(f"\rbatch {i+1}", end="")

        fea = batch[0]
        label = batch[1]
        meta = batch[2]

        # Predict
        preds = model.predict(fea)
        # preds = np.around(preds, 3)
        if (np.ndim(np.squeeze(preds)) > 1) and (abs(preds.sum(axis=1).mean() - 1) > 0.05):
            preds = tf.nn.softmax(preds, axis=1).numpy()
        y_pred_prob.append(preds)
        preds = np.squeeze(preds)

        # If batch size is 1, np.squeeze will create an array of dim [0, 0]
        if np.ndim(preds) == 0:
            preds = [np.asscalar(preds)]

        # Predictions
        if np.ndim(preds) > 1:
            # probabilities (post softmax)
            y_pred_label.extend( np.argmax(preds, axis=1).tolist() )  # SparseCategoricalCrossentropy
        else:
            # p = 0.5
            y_pred_label.extend( [0 if ii < p else 1 for ii in preds] )  # BinaryCrossentropy

        # True labels
        # y_true.extend( label[args.target[0]].numpy().tolist() )  # when batch[1] is dict
        y_true.extend( label.numpy().tolist() )  # when batch[1] is array

        # Meta
        # smp_list.extend( [smp_bytes.decode('utf-8') for smp_bytes in batch[2].numpy().tolist()] )
        for k in meta_keys:
            # print(len(meta[k]))  # the size should as the batch size
            vv = [val_bytes.decode("utf-8") for val_bytes in meta[k].numpy().tolist()]
            if meta_agg[k] is None:
                meta_agg[k] = vv
            else:
                meta_agg[k].extend(vv)

        del batch, fea, label, meta

    # Meta
    df_meta = pd.DataFrame(meta_agg)
    df_meta = df_meta.astype({"tile_id": int}) # "image_id": int
    # print("\ndf memory {:.2f} GB".format( df_meta.memory_usage().sum()/1e9 ))

    # Predictions
    y_pred_prob = np.vstack(y_pred_prob)
    if np.ndim(np.squeeze(y_pred_prob)) > 1:
        # Multiclass classifier
        df_y_pred_prob = pd.DataFrame(y_pred_prob, columns=[f"prob_{c}" for c in range(y_pred_prob.shape[1])])
        y_pred_prob_true = [row[1].values[y] for row, y in zip(df_y_pred_prob.iterrows(), y_true)]
        df_y_pred_prob["prob"] = y_pred_prob_true  # predicted prob of the true class (true_prob)
    else:
        # Binary classifier
        df_y_pred_prob = pd.DataFrame(y_pred_prob, columns=["prob"])

    # True labels
    df_labels = pd.DataFrame({"y_true": y_true, "y_pred_label": y_pred_label})

    # Combine
    prd = pd.concat([df_meta, df_y_pred_prob, df_labels], axis=1)
    # prd = prd.sort_values(split_on, ascending=True)  # split_on is not available here (merged later)
    return prd


def agg_tile_preds(prd, agg_by, meta, agg_method="mean"):
    """ Aggregate tile predictions per agg_by. """
    n_rows = prd.shape[0]
    unq_items = meta[agg_by].nunique()

    if agg_by not in prd.columns:
        prd = meta[[agg_by, "smp"]].merge(prd, on="smp", how="inner")  # assert on shape
        assert prd.shape[0] == n_rows, "Mismatch in number of rows after merge."

    # Agg tile pred on agg_by
    agg_preds = prd.groupby(agg_by).agg({"prob": agg_method}).reset_index()
    agg_preds = prd.groupby(agg_by).agg({"prob": agg_method, "y_true": "unique", "y_pred_label": "unique"}).reset_index()
    # agg_preds = agg_preds.rename(columns={"prob": f"prob_mean_by_{agg_by}"})

    # Merge with meta
    mm = meta.merge(agg_preds, on=agg_by, how="inner")
    mm = mm.drop_duplicates(subset=[agg_by, "Response"])
    assert mm.shape[0] == unq_items, "Mismatch in the number of rows after merge."

    """
    # Efficient use of groupby().apply() !!
    xx = prd.groupby("smp").apply(lambda x: pd.Series({
        "y_true": x["y_true"].unique()[0],
        "y_pred_label": np.argmax(np.bincount(x["y_pred_label"])),
        "pred_acc": sum(x["y_true"] == x["y_pred_label"])/x.shape[0]
    })).reset_index().sort_values(agg_by).reset_index(drop=True)
    xx = xx.astype({"y_true": int, "y_pred_label": int})
    print(agg_preds.equals(xx))
    """

    return mm


def calc_tf_preds(tf_data, meta, model, outdir, args, name, p=0.5, print_fn=print):
    """ ... """
    # Predictions per tile
    # timer = Timer()
    tile_preds = calc_tile_preds(tf_data, model=model, outdir=outdir)
    print_fn("")
    # timer.display_timer(print_fn)

    # Aggregate predictions
    # import ipdb; ipdb.set_trace()
    agg_method = "mean"
    smp_preds = agg_tile_preds(tile_preds, agg_by="smp", meta=meta, agg_method=agg_method)
    grp_preds = agg_tile_preds(tile_preds, agg_by="Group", meta=meta, agg_method=agg_method)

    # Save predictions
    tile_preds.to_csv(outdir/f"{name}_tile_preds.csv", index=False)
    smp_preds.to_csv(outdir/f"{name}_smp_preds.csv", index=False)
    grp_preds.to_csv(outdir/f"{name}_grp_preds.csv", index=False)

    # Scores
    tile_scores = calc_scores(tile_preds["y_true"].values, tile_preds["prob"].values, mltype="cls")
    smp_scores = calc_scores(smp_preds["Response"].values, smp_preds["prob"].values, mltype="cls")
    grp_scores = calc_scores(grp_preds["Response"].values, grp_preds["prob"].values, mltype="cls")

    # dump_dict(tile_scores, outdir/f"{name}_tile_scores.txt")
    # dump_dict(smp_scores, outdir/f"{name}_smp_scores.txt")
    # dump_dict(grp_scores, outdir/f"{name}_grp_scores.txt")

    # Create single scores.csv
    tile_scores["pred_for"] = "tile"
    smp_scores["pred_for"] = "smp"
    grp_scores["pred_for"] = "Group"
    df_scores = pd.DataFrame([tile_scores, smp_scores, grp_scores])
    # df_scores = df_scores[["pred_for"] + sorted([c for c in df_scores.columns if c != "pred_for"])]
    # df_scores = df_scores[["pred_for", "brier", "f1_score", "mcc", "pr_auc", "precision", "recall", "roc_auc"]]
    df_scores = df_scores[["pred_for",
                           "ap_macro", "ap_macro", "ap_weighted",
                           "brier", "f1_score", "mcc", "pr_auc",
                           "precision", "recall", "roc_auc"]]
    df_scores = df_scores.T.reset_index()
    df_scores.columns = df_scores.iloc[0, :]
    df_scores = df_scores.iloc[1:, :]
    df_scores.to_csv(outdir/f"{name}_scores.csv", index=False)

    # Confusion
    print_fn("\n{}".format(yellow("Per-tile confusion:")))
    tile_cnf_mtrx = confusion_matrix(tile_preds["y_true"], tile_preds["y_pred_label"])
    print_fn(tile_cnf_mtrx)
    save_confusion_matrix(true_labels=tile_preds["y_true"].values,
                          predictions=tile_preds["prob"].values,
                          p=p,
                          labels=["Non-response", "Response"],
                          outpath=outdir/f"{name}_tile_confusion.png")

    print_fn("\n{}".format(yellow("Per-sample confusion:")))
    smp_cnf_mtrx = confusion_matrix(smp_preds["Response"], smp_preds["prob"] > p)
    print_fn(smp_cnf_mtrx)
    save_confusion_matrix(true_labels=smp_preds["Response"].values,
                          predictions=smp_preds["prob"].values,
                          labels=["Non-response", "Response"],
                          outpath=outdir/f"{name}_smp_confusion.png")

    print_fn("\n{}".format(yellow("Per-group confusion:")))
    grp_cnf_mtrx = confusion_matrix(grp_preds["Response"], grp_preds["prob"] > p)
    print_fn(grp_cnf_mtrx)
    save_confusion_matrix(true_labels=grp_preds["Response"].values,
                          predictions=grp_preds["prob"].values,
                          labels=["Non-response", "Response"],
                          outpath=outdir/f"{name}_grp_confusion.png")

    print_fn("\n{}".format(cyan("Scores:")))
    print_fn(df_scores)

    return None


def calc_smp_preds(xdata, meta, model, outdir, name, p=0.5, print_fn=print):
    """ Calc predictions using a model that (regular) tabular data (not tf.data).
    Args:
        xdata : pd.DataFrame or np.array
    """
    # Predict
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(xdata)
    else:
        preds = model.predict(xdata)
    # preds = np.around(preds, 3)
    preds = np.squeeze(preds)

    # import ipdb; ipdb.set_trace()
    if np.ndim(preds) > 1:
        # cross-entropy
        y_pred_label = np.argmax(preds, axis=1)
    else:
        # binary cross-entropy
        # p = 0.5
        y_pred_label = [0 if ii < p else 1 for ii in preds]

    # Predictions
    y_pred_prob = preds
    if np.ndim(np.squeeze(y_pred_prob)) == 1:
        # Binary
        df_y_pred_prob = pd.DataFrame(y_pred_prob, columns=["prob"])
    elif np.squeeze(y_pred_prob).shape[1] == 2:
        # Binary
        y_pred_prob = y_pred_prob[:, 1]
        df_y_pred_prob = pd.DataFrame(y_pred_prob, columns=["prob"])
    elif np.squeeze(y_pred_prob).shape[1] > 2:
        # Multiclass
        df_y_pred_prob = pd.DataFrame(y_pred_prob, columns=[f"prob_{c}" for c in range(y_pred_prob.shape[1])])
    else:
        raise ValueError("what's going on with the dim of 'preds'?")

    # True labels
    # y_true = yte["Response"].values
    y_true = meta["Response"].values
    df_labels = pd.DataFrame({"y_true": y_true, "y_pred_label": y_pred_label})

    # -------------------
    # Per-sample analysis
    # -------------------
    # Combine
    prd = pd.concat([meta, df_y_pred_prob, df_labels], axis=1)
    # prd = prd.sort_values(split_on, ascending=True)

    # Save predictions
    prd.to_csv(outdir/f"{name}_smp_preds.csv", index=False)

    # Scores
    smp_scores = calc_scores(prd["y_true"].values, prd["prob"].values, mltype="cls")
    dump_dict(smp_scores, outdir/f"{name}_smp_scores.txt")

    # Confusion
    print_fn("{}".format(yellow("Per-sample confusion:")))
    cnf_mtrx = confusion_matrix(y_true, y_pred_label)
    print_fn(cnf_mtrx)
    save_confusion_matrix(true_labels=prd["y_true"].values,
                          predictions=prd["prob"].values,
                          p=p,
                          labels=["Non-response", "Response"],
                          outpath=outdir/f"{name}_smp_confusion.png")

    # ------------------
    # Per-group analysis
    # ------------------
    grp_prd = prd.groupby("Group").agg({"prob": "mean"}).reset_index()
    # jj = prd[["Sample", "image_id", "Drug1", "Drug2", "trt", "aug", "Group", "grp_name", "Response", "y_true", "y_pred_label"]]
    jj = prd[["Sample", "image_id", "Drug1", "Drug2", "trt", "aug", "Group", "grp_name", "Response", "y_true"]]
    jj = jj.sort_values("Group").reset_index(drop=True)
    df = grp_prd.merge(jj, on="Group", how="inner")
    df["y_pred_label"] = df["prob"].map(lambda x: 0 if x < p else 1)
    df = df.sort_values(["aug", "Group"], ascending=False)
    df = df.drop_duplicates(subset=["Group", "prob"])

    # Scores
    grp_scores = calc_scores(df["y_true"].values, df["prob"].values, mltype="cls")
    dump_dict(grp_scores, outdir/f"{name}_grp_scores.txt")

    # Confusion
    print_fn("\n{}".format(yellow("Per-group confusion:")))
    cnf_mtrx = confusion_matrix(df["y_true"].values, df["y_pred_label"].values)
    print_fn(cnf_mtrx)
    save_confusion_matrix(true_labels=df["y_true"].values,
                          predictions=df["prob"].values,
                          p=p,
                          labels=["Non-response", "Response"],
                          outpath=outdir/f"{name}_grp_confusion.png")

    # ------------------
    # Combined
    # ------------------
    df_smp_scores = pd.DataFrame.from_dict(smp_scores, orient="index", columns=["smp"])
    df_grp_scores = pd.DataFrame.from_dict(grp_scores, orient="index", columns=["Group"])
    scores = pd.concat([df_smp_scores, df_grp_scores], axis=1)
    scores = scores.reset_index().rename(columns={"index": "metric"})
    scores.to_csv(outdir/f"{name}_scores.csv", index=False)
    print_fn("\n{}".format(cyan("Scores:")))
    print_fn(scores)

    return None



# def build_model_rna(pooling='max', pretrain='imagenet'):
#     # Image layers
#     image_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
#     tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")
#     base_img_model = tf.keras.applications.Xception(
#         weights=pretrain, pooling=pooling, include_top=False,
#         input_shape=None, input_tensor=None)
#     x_tile = base_img_model(tile_input_tensor)

#     # RNA layers
#     ge_input_tensor = tf.keras.Input(shape=(976,), name="ge_data")
#     x_ge = Dense(512, activation=tf.nn.relu)(ge_input_tensor)

#     model_inputs = [tile_input_tensor, ge_input_tensor]

#     # Merge towers
#     merged_model = layers.Concatenate(axis=1, name="merger")([x_ge, x_tile])

#     hidden_layer_width = 1000
#     merged_model = tf.keras.layers.Dense(hidden_layer_width, activation=tf.nn.relu,
#                                          name="hidden_1")(merged_model)

#     # Add the softmax prediction layer
#     activation = 'linear' if model_type == 'linear' else 'softmax'
#     final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
#     softmax_output = tf.keras.layers.Activation(activation, dtype='float32', name='ctype')(final_dense_layer)

#     # Assemble final model
#     model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)
#     return model
