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

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

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

DATADIR = fdir/'../data'
TILES_DIR = DATADIR/'tiles_png'
TFR_DIR = DATADIR/'tfrecords'
os.makedirs(TFR_DIR, exist_ok=True)


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate tfrecords for training.')

    parser.add_argument('-ns', '--n_samples',
                        default=None,
                        type=int,
                        help='Number of samples (treatments) to process (default: 1).')

    # args = parser.parse_args(args)
    args, other_args = parser.parse_known_args()
    return args


# def calc_records_in_tfr_folder(tfr_dir):
#     """
#     Calc and print the number of examples (tiles) in all tfrecords in the
#     input folder.
#     """
#     count = 0
#     for tfr_path in sorted(tfr_dir.glob('*.tfrec*')):
#         count += sum(1 for _ in tf.data.TFRecordDataset(str(tfr_path)))
#     print('Number of examples in all tfrecords in the folder:', count)


# def calc_examples_in_tfrecord(tfr_path):
#     """
#     Calc and print the number of examples (tiles) in the input tfrecord
#     file provided by the path to the file.
#     """
#     count = sum(1 for _ in tf.data.TFRecordDataset(str(tfr_path)))
#     print('Number of examples in the tfrecord:', count)


def show_img(img, title=None):
    """ Show a single image tile. """
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_images(img_list, ncols=4):
    """ Show a few image tiles. """
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 20))

    for i, img_id in enumerate(np.random.randint(0, len(img_list), ncols)):
        ax[i].imshow(img_list[img_id]['image']);
        ax[i].axis("off");
        ax[i].set_title(img_list[img_id]['slide'])


def encode_categorical(df, label_name, label_value):
    """
    The label_name and label_value are columns in df which, respectively,
    correspond to the name and value of a categorical variable.
    
    Args:
        label_name:  name of the label
        label_value: numerical value assigned to the label
    Returns:
        dict of unique label names the appropriate values {label_name: label_value}
    """
    aa = df[[label_name, label_value]].drop_duplicates().sort_values(label_value).reset_index(drop=True)
    return dict(zip(aa[label_name], aa[label_value]))


def load_data(path):
    """ Load dataframe that contains tabular features including rna and descriptors
    (predixed with ge_ and dd_), metadata, and response.
    """
    data = pd.read_csv(path)
    csite_enc = encode_categorical(df=data, label_name='csite', label_value='csite_label')
    ctype_enc = encode_categorical(df=data, label_name='ctype', label_value='ctype_label')
    CSITE_NUM_CLASSES = len(csite_enc.keys())
    CTYPE_NUM_CLASSES = len(ctype_enc.keys())
    
    return data


def _float_feature(value):
    """ Returns a bytes_list from a float / double. """
    if isinstance(value, list) is False:
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):                                                                                                                                    
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, list) is False:
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))                                                                                     


def _int64_feature(value):                                                                                                                                    
    """ Returns an int64_list from a bool / enum / int / uint. """
    if isinstance(value, list) is False:
        value = [value]    
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def np_img_to_bytes(np_img):
    """ Encode np image into bytes string. 
    https://programmer.group/opencv-python-cv2.imdecode-and-cv2.imencode-picture-decoding-and-coding.html
    Note! cv2 assumes images are in BGR rather RGB
    """
    _, img_encode = cv2.imencode('.png', cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
    # img_encode = np.array(img_encode)
    img_bytes = img_encode.tobytes()
    return img_bytes


def build_dfrecords(data, n_samples=None):
    """ Build tfrecords using the master dataframe that contains tabular
    features and response values, and image tiles.
    """
    if n_samples is None:
        n_samples = data.shape[0]

    # Iter over treatment samples
    for i, item in itertools.islice(data.iterrows(), n_samples):
    # for i, item in data.iterrows():
        # Prefix rna and drug features
        ge_vec = [value for col_name, value in zip(item.index, item.values) if col_name.startswith('ge_')]
        dd_vec = [value for col_name, value in zip(item.index, item.values) if col_name.startswith('dd_')]

        tfr_fname = TFR_DIR/f'{item.smp}.tfrecord'  # tfrecord filename
        tiles_path_list = sorted(TILES_DIR.glob(f'{item.image_id}-tile*.png'))  # load slide tiles
        tiles_path_list = [str(p) for p in tiles_path_list]

        writer = tf.io.TFRecordWriter(str(tfr_fname))  # create tfr writer

        # Iter over tiles
        for i, tile_path in enumerate(tiles_path_list):
            np_img = slide.open_image_np(str(tile_path), verbose=False)
            img_bytes = np_img_to_bytes(np_img)

            def parse_tile_fpath(path):
                """ Extract the basename from the tile fpath and return a list
                of the parts that construct the original fpath. """
                p = os.path.basename(str(path))
                p = p.split('.png')[0]
                p = p.split('-')
                return p

            tile_path_parts = parse_tile_fpath(tile_path)
            tile_id = bytes(str(item['image_id']) + '_' + str(i+1), 'utf-8')

            # Prep features for current example to be assigned into the tfrecord
            feature = {
                'ge_vec': _float_feature(ge_vec),
                'dd_vec': _float_feature(dd_vec),

                'smp': _bytes_feature(bytes(item['smp'], 'utf-8')),
                'Response': _int64_feature(item['Response']),
                'image_id': _int64_feature(item['image_id']),

                # tile
                'image_raw': _bytes_feature(img_bytes),

                # extract meta from tile filename
                'tile_id': _bytes_feature(tile_id),
                'row': _bytes_feature(bytes(tile_path_parts[2], 'utf-8')),
                'col': _bytes_feature(bytes(tile_path_parts[3], 'utf-8')),

                'Sample': _bytes_feature(bytes(item['Sample'], 'utf-8')),
                'ctype': _bytes_feature(bytes(item['ctype'], 'utf-8')),
                'csite': _bytes_feature(bytes(item['csite'], 'utf-8')),
                'ctype_label': _int64_feature(item['ctype_label']),
                'csite_label': _int64_feature(item['csite_label']),
            }
            ex = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(ex.SerializeToString())

        writer.close()
    return


def run(args):

    # import ipdb; ipdb.set_trace(context=11)

    # Read dataframe
    data = load_data(DATADIR/'data_merged.csv')
    GE_LEN = sum([1 for c in data.columns if c.startswith('ge_')])
    DD_LEN = sum([1 for c in data.columns if c.startswith('dd_')])

    # ------------------------------
    # Build TFRecords
    # ------------------------------
    build_dfrecords(data, n_samples=args.n_samples)
    print('\nFinished building tfrecords.')

    # Summary of tfrecords
    tfr_files = sorted(TFR_DIR.glob('*.tfrec*'))
    print('\nNumber of tfrecords the folder:', len(tfr_files))
    calc_records_in_tfr_folder(tfr_dir=TFR_DIR)
    calc_examples_in_tfrecord(tfr_path=str(tfr_files[0]))

    # ------------------------------------------
    # Read a tfrecord and explore single example
    # ------------------------------------------
    # Feature specs (used to read an example from tfrecord)
    fea_spec = {
        'ge_vec': tf.io.FixedLenFeature(shape=(GE_LEN,), dtype=tf.float32, default_value=None),
        'dd_vec': tf.io.FixedLenFeature(shape=(DD_LEN,), dtype=tf.float32, default_value=None),

        'smp':      tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'Response': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
        'image_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),

        'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),

        'tile_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'row':     tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'col':     tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),

        'Sample': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'ctype':  tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'csite':  tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'ctype_label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
        'csite_label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    }

    # Create tf dataset object (from a random tfrecord)
    ds = tf.data.TFRecordDataset(str(tfr_files[0]))

    # Get single tfr example
    ex = next(ds.__iter__())
    ex = tf.io.parse_single_example(ex, features=fea_spec)  # returns features of the example in a dict
    print('\nFeature types in the tfrecord example:\n{}'.format(ex.keys()))

    # Bytes
    print('\nBytes (tile_id):')
    print(ex['tile_id'].numpy())
    # print(ex['tile_id'].numpy().decode('UTF-8'))

    # Bytes
    print('\nBytes (csite):')
    print(ex['csite'].numpy())
    # print(ex['csite'].numpy().decode('UTF-8'))

    # Float
    print('\nFloat (csite_label):')
    print(ex['csite_label'])
    # print(ex['csite_label'].numpy())

    # Int
    print('\nInt (Response):')
    print(ex['Response'])
    # print(ex['Response'].numpy())

    # Float
    print('\nFloat (ge_vec):')
    print(ex['ge_vec'][:10])
    # print(ex['ge_vec'].numpy()[:10])

    # Bytes
    print('\nBytes (image_raw):')
    img = tf.image.decode_jpeg(ex['image_raw'], channels=3)
    print(img.numpy().shape)
    # show_img(img)


def main(args):
    t = util.Time()
    args = parse_args(args)
    run(args)
    t.elapsed_display()
    print('Done.')


if __name__ == "__main__":
    main(sys.argv[1:])
