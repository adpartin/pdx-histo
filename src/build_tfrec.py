import os
import sys
assert sys.version_info >= (3, 5)

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

# Seed
np.random.seed(42)


dirpath = Path(__file__).resolve().parent


def calc_records_in_tfr_folder(tfr_dir):
    """
    Calc and print the number of examples (tiles) in all tfrecords in the
    input folder.
    """
    count = 0
    for tfr_path in sorted(tfr_dir.glob('*.tfrec*')):
        count += sum(1 for _ in tf.data.TFRecordDataset(str(tfr_path)))
    print('Number of examples in all tfrecords in the folder:', count)

    
def calc_examples_in_tfrecord(tfr_path):
    """
    Calc and print the number of examples (tiles) in the input tfrecord
    file provided by the path to the file.
    """
    count = sum(1 for _ in tf.data.TFRecordDataset(str(tfr_path)))
    print('Number of examples in the tfrecord:', count)

    
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
    
    # Create column of unique treatments
    col_name = 'smp'
    if col_name not in data.columns:
        jj = [str(s) + '_' + str(d) for s, d in zip(data.Sample, data.Drug1)]
        data.insert(loc=0, column=col_name, value=jj, allow_duplicates=False)
        
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


def build_dfrecords(data, tfr_path, tiles_path, n_samples=None):
    """ Build tfrecords using the master dataframe that contains tabular
    features and response values, and image tiles.
    Args:
        tiles_path : path to the tiles

    """
    if n_samples is None:
        n_samples = data.shape[0]

    # Iter over treatment samples
    for i, item in itertools.islice(data.iterrows(), n_samples):
    # for i, item in data.iterrows():
        # Prefix rna and drug features
        ge_vec = [value for col_name, value in zip(item.index, item.values) if col_name.startswith('ge_')]
        dd_vec = [value for col_name, value in zip(item.index, item.values) if col_name.startswith('dd_')]

        tfr_fname = tfr_path/f'train_{item.smp}.tfrecord'  # tfrecord filename
        tiles_path_list = list(tiles_path.glob(f'{item.image_id}-tile*.png'))  # load slide tiles
        writer = tf.io.TFRecordWriter(str(tfr_fname))  # create tfr writer

        # Iter over tiles
        for tile_path in tiles_path_list:
            np_img = slide.open_image_np(str(tile_path), verbose=False)
            img_bytes = np_img_to_bytes(np_img)

            # Prep features for current example to be assigned into the tfrecord
            feature = {
                    'ge_vec': _float_feature(ge_vec),
                    'dd_vec': _float_feature(dd_vec),

                    'smp': _bytes_feature(bytes(item['smp'], 'utf-8')),
                    'rsp': _int64_feature(item['Response']),
                    'image_id': _int64_feature(item['image_id']),

                    # tile
                    'image_raw': _bytes_feature(img_bytes),
                    # TODO tile meta                

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


if __name__ == "__main__":
    
    t = util.Time()
    # import ipdb; ipdb.set_trace(context=11)
    
    # Path
    datapath = dirpath/'../data'
    tiles_path = datapath/'tiles_png'
    tfr_path = datapath/'tfrecords'
    os.makedirs(tfr_path, exist_ok=True)
    
    # Read dataframe
    data = load_data(datapath/'data_merged.csv')
    GE_LEN = sum([1 for c in data.columns if c.startswith('ge_')])
    DD_LEN = sum([1 for c in data.columns if c.startswith('dd_')])
        
    # -----------------------------------------------------------------------------
    # Subsample data to create balanced dataset in terms of drug response and ctype
    # -----------------------------------------------------------------------------
    print('\nSubsample master dataframe to create balanced dataset.')
    r0 = data[data.Response == 0]  # non-responders
    r1 = data[data.Response == 1]  # responders

    dfs = []
    for ctype, count in r1.ctype.value_counts().items():
        # print(ctype, count)
        aa = r0[r0.ctype == ctype]
        if aa.shape[0] > count:
            aa = aa.sample(n=count)
        dfs.append(aa)

    aa = pd.concat(dfs, axis=0)
    df = pd.concat([aa, r1], axis=0).reset_index(drop=True)
    print(df.shape)

    aa = df.reset_index()
    pprint(aa.groupby(['ctype', 'Response']).agg({'index': 'nunique'}).reset_index())

    data = df
    del dfs, df, aa, ctype, count
    
    # -------------------------------------
    # Copy slides to training_slides folder
    # -------------------------------------
    print('\nCopy slides to training_slides folder.')
    src_img_path = datapath/'doe-globus-pdx-data'
    dst_img_path = datapath/'training_slides'
    os.makedirs(dst_img_path, exist_ok=True)

    exist = []
    copied = []
    for fname in data.image_id.unique():
        if (dst_img_path/f'{fname}.svs').exists():
            exist.append(fname)
        else:
            _ = shutil.copyfile(str(src_img_path/f'{fname}.svs'), str(dst_img_path/f'{fname}.svs'))
            copied.append(fname)

    print(f'Copied slides:   {len(copied)}')
    print(f'Existing slides: {len(exist)}')
    
    # ------------------------------
    # Build TFRecords
    # ------------------------------
    n_samples = 1
    build_dfrecords(data, tfr_path=tfr_path, tiles_path=tiles_path, n_samples=n_samples)
    print('\nFinished building tfrecords.')
        
    # Glob tfrecords
    tfr_files = sorted(tfr_path.glob('*.tfrec*'))
    print('\nNumber of tfrecords the folder:', len(tfr_files))
    calc_records_in_tfr_folder(tfr_dir=tfr_path)
    calc_examples_in_tfrecord(tfr_path=str(tfr_files[0]))
    
    # -----------------------------------------------
    # Read single tfrecord and explore single example
    # -----------------------------------------------
    # Feature specs (used to read an example from tfrecord)
    fea_spec = {
        'ge_vec': tf.io.FixedLenFeature(shape=(GE_LEN,), dtype=tf.float32, default_value=None),
        'dd_vec': tf.io.FixedLenFeature(shape=(DD_LEN,), dtype=tf.float32, default_value=None),    

        'smp':      tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None), 
        'rsp': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 
        'image_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None), 

        'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),

        'Sample': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'ctype':  tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
        'csite':  tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),    
        'ctype_label':  tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
        'csite_label':  tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    }

    # Create tf dataset object
    # fname = 'train_135848~042-T~JI6_NSC.616348.tfrecord'
    fname = 'train_165739~295-R~AM1_NSC.616348.tfrecord'
    ds = tf.data.TFRecordDataset(str(tfr_path/fname))

    # Get single tfr example
    ex = next(ds.__iter__())
    ex = tf.io.parse_single_example(ex, features=fea_spec)  # returns the features for a given example in a dict, ex_fea_dct
    print('Feature types in the tfrecord example:\n{}'.format(ex.keys()))

    # Bytes
    print('\nBytes (csite):')
    print(ex['csite'].numpy())
    print(ex['csite'].numpy().decode('UTF-8'))

    # Float
    print('\nFloat (csite_label):')
    print(ex['csite_label'])
    print(ex['csite_label'].numpy())

    # Int
    print('\nInt (rsp):')
    print(ex['rsp'])
    print(ex['rsp'].numpy())

    # Float
    print('\nFloat (ge_vec):')
    print(ex['ge_vec'][:10])
    print(ex['ge_vec'].numpy()[:10])

    # Bytes
    print('\nBytes (image_raw):')
    img = tf.image.decode_jpeg(ex['image_raw'], channels=3)
    print(img.numpy().shape)
    # show_img(img)
    
    t.elapsed_display()
    print('Done.')
    
    
    