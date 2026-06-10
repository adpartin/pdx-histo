from pathlib import Path
from time import time
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
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Concatenate
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


class MySparseBCE_From_Logits(losses.Loss):
    """ ... """
    def __init__(self, class_weight=[1, 1], dtype=tf.float32):
        super().__init__()
        self.class_weight = class_weight
        self.from_logits = True
        self.dtype = dtype

    def call(self, y_true, y_pred):
        labels = tf.cast(y_true, self.dtype)
        logits = tf.cast(y_pred, self.dtype)
        if self.from_logits:
            probs = tf.cast(tf.math.sigmoid(logits), self.dtype)

        # # Onehot labels
        # n_classes = 2
        # onehot_labels = tf.one_hot(ytr_batch, depth=n_classes, on_value=1)

        # Vector of weights
        # weight = tf.gather(params=[1, 10], indices=tf.cast(labels, tf.int32))
        weight = tf.gather(params=list(self.class_weight.values()), indices=tf.cast(labels, tf.int32))
        weight = tf.cast(weight, self.dtype)

        # Keras BCE loss
        # keras_bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=self.from_logits)
        # print("Keras BCE loss: {}".format(keras_bce_loss(labels, logits)))

        # Manual BCE loss
        # impl of cross-entropy: https://stackoverflow.com/questions/58159154
        # nans in loss func: https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
        # weighted_bce_losses = -(weight*labels*(tf.math.log(probs)) + weight*(1 - labels)*(tf.math.log(1 - probs)))
        p0 = weight * labels * tf.math.log(tf.clip_by_value(probs, 1e-10, 1.0))
        p1 = weight * (1 - labels) * tf.math.log(tf.clip_by_value(1 - probs, 1e-10, 1.0))
        weighted_bce_losses = -(p0 + p1)
        # weighted_bce_loss = tf.reduce_mean(weighted_bce_losses)  # reduction
        # print("Manual weigted BCE loss: {}".format(weighted_bce_loss))
        return weighted_bce_losses


# ------------------------------------------------------------------
class Multimodal():
    """ ... """
    def __init__(self, print_fn=print):
        super(Multimodal, self).__init__()
        self.model = None
        self.print_fn = print_fn

    @tf.function
    def train_step(self, xtr_batch, ytr_batch):
        """ Training step. """
        ytr_batch = tf.squeeze(ytr_batch)
        with tf.GradientTape() as tape:
            # Forward pass and loss
            logits = self.model(xtr_batch, training=True)
            logits = tf.squeeze(logits)
            loss_value = self.loss_fn(ytr_batch, logits)
        # Calc the grads and update the params
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # Update training metric
        probs = tf.math.sigmoid(logits)
        self.trn_roc_met.update_state(ytr_batch, probs)
        self.trn_prc_met.update_state(ytr_batch, probs)
        return loss_value

    @tf.function
    def val_step(self, xvl_batch, yvl_batch):
        """ Validation/test step. """
        yvl_batch = tf.squeeze(yvl_batch)
        val_logits = self.model(xvl_batch, training=False)
        val_logits = tf.squeeze(val_logits)
        val_loss = self.loss_fn(yvl_batch, val_logits)
        val_probs = tf.math.sigmoid(val_logits)
        self.val_roc_met.update_state(yvl_batch, val_probs)
        self.val_prc_met.update_state(yvl_batch, val_probs)
        return val_loss

    def evaluate(self):
        """ Run a validation loop and return metrics. """
        for vl_step, (xvl_batch, yvl_batch) in enumerate(self.validation_data):
            vl_step += 1
            if vl_step > self.validation_steps:
                break
            val_loss = self.val_step(xvl_batch, yvl_batch)

        # Display metrics at the end of each epoch
        val_roc = self.val_roc_met.result().numpy()
        val_prc = self.val_prc_met.result().numpy()

        # Reset metrics at the end of each epoch
        self.val_roc_met.reset_states()
        self.val_prc_met.reset_states()

        res = {"val_loss": val_loss.numpy(), "val_roc": val_roc, "val_prc": val_prc}
        return res

    def print_trn_scores(self):
        # Display metrics at the end of each epoch
        trn_roc = self.trn_roc_met.result().numpy()
        trn_prc = self.trn_prc_met.result().numpy()
        val_roc = self.val_roc_met.result().numpy()
        val_prc = self.val_prc_met.result().numpy()
        self.print_fn("epoch {}, loss: {:.3f}, roc: {:.3f}, "
                      "prc: {:.3f}, val_loss: {:.3f}, val_roc: {:.3f} val_prc: {:.3f}".format(
                          epoch, loss_value, trn_roc, trn_prc,
                          evals["val_loss"], evals["val_roc"], evals["val_prc"]))

    def set_optimizer(self, optimizer_name, learning_rate):
        if optimizer_name == "SGD":
            self.optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        elif optimizer_name == "Adam":
            self.optimizer = optimizers.Adam(learning_rate=learning_rate)

    def myfit(self,
              train_data,
              validation_data, 
              steps_per_epoch, 
              validation_steps, 
              epochs,
              batch_patience: int=100, 
              validate_on_batch=250, 
              min_epochs: int=4,
              loss_fn=losses.BinaryCrossentropy(from_logits=True),
              optimizer_name="Adam", 
              learning_rate=0.0005,
              outdir=Path("."),
              verbose=0):
        """ 
        Args:
            batch_patience : number of batches with no improvement after which training will be stopped
            min_epochs : min number of epochs to train the model
        """
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.steps_per_epoch = steps_per_epoch
        self.batch_patience = batch_patience
        self.best = np.Inf      # init the best as infinity
        self.epoch = 0
        self.min_epochs = min_epochs
        self.outdir = outdir
        self.stopped_epoch = 0  # epoch the training stops at
        self.stopped_batch = 0  # batch the training stops at
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.val_loss = np.Inf
        self.validate_on_batch = validate_on_batch

        self.trn_roc_met = keras.metrics.AUC(name="roc-auc", curve="ROC")
        self.val_roc_met = keras.metrics.AUC(name="roc-auc", curve="ROC")
        self.trn_prc_met = keras.metrics.AUC(name="prc-auc", curve="PR")
        self.val_prc_met = keras.metrics.AUC(name="prc-auc", curve="PR")

        assert self.model is not None, "Model is not defined."

        self.set_optimizer(optimizer_name, learning_rate)

        # Iter over epochs
        for epoch in range(self.epochs):
            t0 = time()
            epoch += 1  # inc epoch counter
            wait = 0

            # Iter over steps (batches)
            for step, (xtr_batch, ytr_batch) in enumerate(train_data):
                step += 1  # inc step counter
                if step > steps_per_epoch:
                    break
                loss_value = self.train_step(xtr_batch, ytr_batch)

                # print("\repoch {}/{}, step {}/{}, loss: {:.5f}".format(
                #     epoch, self.epochs, step, steps_per_epoch, loss_value), end="\r")

                if step % self.validate_on_batch == 0:
                    evals = self.evaluate()
                    current = evals["val_loss"]

                    # Here we consider only error metrics (the lower the better)
                    if np.less(current, self.best):
                        self.best = current
                        wait = 0
                        self.best_weights = self.model.get_weights()
                        # Save model
                    else:
                        wait += 1

                    # Don't terminate on the first epoch
                    if (wait >= self.batch_patience) and (epoch > self.min_epochs):
                        self.stopped_epoch = epoch
                        self.stopped_batch = step
                        self.print_fn("\n{}".format(red(f"Early stop (terminated training at epoch: {epoch}, step: {step}).")))
                        self.print_fn("Restores model weights from the best [epoch, batch] combination.")
                        self.model.set_weights(self.best_weights)
                        evals = self.evaluate()
                        self.print_fn("epoch {}, loss: {:.5f}, roc: {:.3f}, "
                                      "prc: {:.3f}, val_loss: {:.5f}, val_roc: {:.3f} val_prc: {:.3f}".format(
                                          epoch, loss_value, trn_roc, trn_prc,
                                          evals["val_loss"], evals["val_roc"], evals["val_prc"]))
                        self.print_fn("Saves best model.")
                        self.model.save(self.outdir/"best_model.ckpt")
                        return None

                    # Log metrics after evaluation
                    print("\repoch {}, step {}/{}, loss: {:.5f}, val_loss: {:.5f}, best_val_loss: {:.5f} (wait: {})".format(
                        epoch, step, steps_per_epoch, loss_value, evals["val_loss"], self.best, yellow(wait)), end="\r")

            # Display metrics at the end of each epoch
            trn_roc = self.trn_roc_met.result().numpy()
            trn_prc = self.trn_prc_met.result().numpy()

            # Reset metrics at the end of each epoch
            self.trn_roc_met.reset_states()
            self.trn_prc_met.reset_states()

            # Run a validation loop at the end of each epoch
            evals = self.evaluate()

            tm = (time() - t0)/60
            self.print_fn("epoch {} ({:.1f} min), loss: {:.5f}, roc: {:.3f}, "
                          "prc: {:.3f}, val_loss: {:.5f}, val_roc: {:.3f} val_prc: {:.3f}".format(
                              epoch, tm, loss_value, trn_roc, trn_prc,
                              evals["val_loss"], evals["val_roc"], evals["val_prc"]))

        self.print_fn("\n{}".format(red(f"Completed training (finished {epoch} epochs and {step} steps).")))
        self.print_fn("Restores model weights from the best epoch-batch set.")
        self.model.set_weights(self.best_weights)
        self.model.save(self.outdir/"best_model.ckpt")
        return None

    def build_model_rsp(self,
                        use_ge=True,
                        use_dd1=True,
                        use_dd2=True,
                        use_tile=True,
                        ge_shape=None,
                        dd_shape=None,
                        dense1_img=1024,
                        dense2_img=512,
                        dense1_ge=500,
                        dense1_dd1=250,
                        dense1_dd2=250,
                        dense1_top=1000,
                        dropout1_top=0.1,
                        output_bias=None,
                        base_image_model="Xception",
                        pooling="avg",
                        pretrain="imagenet",
                        loss_fn=losses.BinaryCrossentropy(),
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

        # Dropout for feature type
        # ModalDropout
        pass

        # Merge towers
        merged_model = Concatenate(axis=1, name="merger")(merge_inputs)

        # Dense layers of the top classfier
        merged_model = Dense(dense1_top, activation=tf.nn.relu, name="dense1_top")(merged_model)
        merged_model = BatchNormalization(name="batchnorm_top")(merged_model)
        if dropout1_top > 0:
            merged_model = Dropout(dropout1_top)(merged_model)

        output = Dense(1, name="logits")(merged_model)
        if from_logits is False:
            output = Activation(tf.nn.sigmoid, name="Response")(output)

        # Assemble final model
        model = Model(inputs=model_inputs, outputs=output)

        self.model = model
        return None
    # ------------------------------------------------------------------


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
                    loss_fn=losses.BinaryCrossentropy(),
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
    if from_logits is False:
        output = Activation(tf.nn.sigmoid, name="Response")(output)

    # Assemble final model
    model = Model(inputs=model_inputs, outputs=output)

    # These metrics don't work with logits
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="BinAcc")
          # keras.metrics.FalsePositives(name="fp"),
          # keras.metrics.TruePositives(name="tp"),
          # keras.metrics.AUC(name="roc-auc", curve="ROC"),
          # keras.metrics.AUC(name="prc-auc", curve="PR"),
    ]

    if optimizer == "SGD":
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "Adam":
        optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
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
            # multiclass
            preds = tf.nn.softmax(preds, axis=1).numpy()
        if (np.ndim(np.squeeze(preds)) == 1) and (abs(preds).max() > 1.0) or (abs(preds).max() < 0.0):
            # binary
            preds = tf.nn.sigmoid(preds).numpy()
        y_pred_prob.append(preds)
        preds = np.squeeze(preds)

        # If batch size is 1, np.squeeze will create an array of dim [0, 0]. Fixed with this.
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


def focal_loss(gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed
