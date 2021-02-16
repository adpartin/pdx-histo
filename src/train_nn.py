import os
import sys
assert sys.version_info >= (3, 5)

import argparse
from pathlib import Path
from pprint import pprint
import glob
import shutil
import itertools
import pandas as pd
import numpy as np
from typing import List

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)

import cv2
import openslide
# from deephistopath.wsi import filter
from deephistopath.wsi import slide
# from deephistopath.wsi import tiles
from deephistopath.wsi import util

from tf_utils import calc_records_in_tfr_folder, calc_examples_in_tfrecord

# Seed
np.random.seed(42)


fdir = Path(__file__).resolve().parent

# Path
MAIN_APPDIR = fdir/'../apps'
DATADIR = fdir/'../data'
FILENAME = 'annotations.csv'
TILES_DIR = DATADIR/'tiles_png'
TFR_DIR = DATADIR/'tfrecords'


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate tfrecords for training.')

    parser.add_argument('-an', '--appname',
                        default='mm_01',
                        type=str,
                        help='App name to dump the splits.')

    # args = parser.parse_args(args)
    args, other_args = parser.parse_known_args()
    return args


def get_tfr_fnames(smp_names_subset) -> List[str]:
    """ Get filenames of tfrecords based on response samples names.

    Example:
        get_tfr_fnames(tr_smp_names)
    """
    tfr_all_files = sorted(TFR_DIR.glob('*.tfrec*'))
    tfr_sub_files = []

    for sname in smp_names_subset:
        fname = TFR_DIR/(sname+'.tfrecord')
        if fname not in tfr_all_files:
            raise ValueError('File was not found in:\n\t{fname}')
        tfr_sub_files.append(str(fname))

    return tfr_sub_files


def run(args):

    # Load data
    appdir = MAIN_APPDIR/args.appname
    data = pd.read_csv(appdir/FILENAME)
    print(data.shape)
    pprint(data.groupby(['ctype', 'Response']).agg({'smp': 'nunique'}).reset_index().rename(columns={'smp': 'samples'}))

    GE_LEN = sum([1 for c in data.columns if c.startswith('ge_')])
    DD_LEN = sum([1 for c in data.columns if c.startswith('dd_')])

    # Summary of tfrecords
    tfr_all_files = sorted(TFR_DIR.glob('*.tfrec*'))
    print('\nNumber of tfrecords the folder:', len(tfr_all_files))
    # calc_records_in_tfr_folder(tfr_dir=TFR_DIR)
    calc_examples_in_tfrecord(tfr_path=str(tfr_all_files[0]))

    # -----------------------------------------------
    #       Data splits
    # -----------------------------------------------
    # T/V/E filenames
    splitdir = appdir/'splits'
    split_id = 0
    
    split_pattern = f'1fold_s{split_id}_*_id.csv'
    single_split_files = glob.glob(str(splitdir/split_pattern))
    # single_split_files = list(splitdir.glob(split_pattern))

    # Get indices for the split
    # assert len(single_split_files) >= 2, f'The split {s} contains only one file.'
    for id_file in single_split_files:
        if 'tr_id' in id_file:
            tr_id = pd.read_csv(id_file).values.reshape(-1,)
        elif 'vl_id' in id_file:
            vl_id = pd.read_csv(id_file).values.reshape(-1,)
        elif 'te_id' in id_file:
            te_id = pd.read_csv(id_file).values.reshape(-1,)

    cv_lists = (tr_id, vl_id, te_id)

    # -----------------------------------------------
    #       Get data based on splits
    # -----------------------------------------------
    # Dfs of T/V/E samples
    tr_df = data.iloc[tr_id, :].reset_index(drop=True)
    vl_df = data.iloc[vl_id, :].reset_index(drop=True)
    te_df = data.iloc[te_id, :].reset_index(drop=True)

    # List of sample names for T/V/E
    tr_smp_names = tr_df['smp'].values
    vl_smp_names = vl_df['smp'].values
    te_smp_names = te_df['smp'].values

    # x_data = data.iloc[idx, :].reset_index(drop=True)
    # y_data = np.squeeze(ydata.iloc[idx, :]).reset_index(drop=True)

    tr_tfr_fnames = get_tfr_fnames(tr_smp_names)
    vl_tfr_fnames = get_tfr_fnames(vl_smp_names)
    te_tfr_fnames = get_tfr_fnames(te_smp_names)

    # ------------------------------------------
    # Read a tfrecord and explore single example
    # ------------------------------------------

    # Feature specs (used to read an example from tfrecord)
    FEA_SPEC = {
        'ge_vec': tf.io.FixedLenFeature(shape=(GE_LEN,), dtype=tf.float32, default_value=None),
        'dd_vec': tf.io.FixedLenFeature(shape=(DD_LEN,), dtype=tf.float32, default_value=None),    

        'smp':      tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None), 
        'Response': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
        'image_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 

        'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),

        'Sample': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'ctype':  tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'csite':  tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),    
        'ctype_label':  tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
        'csite_label':  tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    }

    # def prepare_image(img, augment=True, dim=256):
    def prepare_image(img, augment=True):
        """
        Prepare single image for training. 
        Deotte.
        www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords/
        """
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0

        if augment:
            # https://www.tensorflow.org/api_docs/python/tf/image/random_flip_up_down
            # img = transform(img, DIM=dim)
            img = tf.image.random_flip_left_right(img)
            #img = tf.image.random_hue(img, 0.01)
            # img = tf.image.random_saturation(img, 0.7, 1.3)
            # img = tf.image.random_contrast(img, 0.8, 1.2)
            # img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_flip_up_down(img)  # ap

        # img = tf.reshape(img, [dim, dim, 3])  
        return img

    # Note! because this func returns inputs and outputs, it's probably must
    # change depending on the prediction task 
    def read_tfr_example(ex, augment=True):
        """ Read and parse a single example from a tfrecord, and prepare
        inputs and outputs for TF model training.
        """
        ex = tf.io.parse_single_example(ex, FEA_SPEC)

        # Image data
        img = ex['image_raw']
        img = prepare_image(img, augment=augment)  # Deotte

        # Features (RNA and descriptors)
        ge_vec = tf.cast(ex['ge_vec'], tf.float32)
        dd_vec = tf.cast(ex['dd_vec'], tf.float32)

        # Dict of multi-input features
        inputs = {'ge_vec': ge_vec, 'dd_vec': dd_vec, 'img': img}

        # Dict of single-output classifier
        rsp = tf.cast(ex['Response'], tf.int64)
        outputs = {'Response': rsp}

        return inputs, outputs
    
    import ipdb; ipdb.set_trace(context=11)

    # Read single example
    ds = tf.data.TFRecordDataset(filenames=tr_tfr_fnames)
    ds = ds.map(lambda ex: read_tfr_example(ex, augment=True))

    # Take an example
    # ex = next(ds.take(count=1).__iter__())  # creates Dataset with at most 'count' elements from this dataset.
    ex = next(ds.__iter__())
    print('Inputs: ', ex[0].keys())
    print('Outputs:', ex[1].keys())

    print('\nInput features:')
    for i, fea_name in enumerate(ex[0].keys()):
        print(fea_name, ex[0][fea_name].numpy().shape)

    print('\nOutputs:')
    for i, out_name in enumerate(ex[1].keys()):
        print(out_name, ex[1][out_name].numpy().shape)

    # ------------------------------------------
    # Define TF Datasets for T/V/E
    # ------------------------------------------        
    ds_tr = tf.data.TFRecordDataset(filenames=tr_tfr_fnames)
    ds_vl = tf.data.TFRecordDataset(filenames=vl_tfr_fnames)
    ds_te = tf.data.TFRecordDataset(filenames=te_tfr_fnames)
    
    
    print('\nTODO ...\n')
    

def main(args):
    timer = util.Time()
    args = parse_args(args)
    run(args)
    timer.elapsed_displa()
    print('Done.')


if __name__ == '__main__':
    main(sys.argv[1:])
