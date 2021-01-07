# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
    !pip install -q -U tfx==0.21.2
    print("You can safely ignore the package incompatibility errors.")
except Exception:
    pass

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf


dirpath = Path(__file__).resolve().parent


def calc_records_in_tfr_folder(tfr_dir):
    """ Calc total number of examples (tiles) in all tfrecords. """
    count = 0
    for tfr_path in sorted(tfr_dir.glob('*.tfrec*')):
        count += sum(1 for _ in tf.data.TFRecordDataset(str(tfr_path)))
    print('Number of examples in all tfrecords in the folder:', count)

    
def calc_examples_in_tfrecord(tfr_path):
    """ Calc total number of examples (tiles) in all tfrecords. """
    count = sum(1 for _ in tf.data.TFRecordDataset(str(tfr_path)))
    print('Number of examples in the tfrecord:', count)

    
def show_img(img, title=None):
    """ Show a single image tile. """
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()
    
    
def show_images(img_list, ncols=4):
    """ Show  single image tile. """
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 20))
    
    for i, img_id in enumerate(np.random.randint(0, len(img_list), ncols)):
        ax[i].imshow(img_list[img_id]['image']); ax[i].axis("off"); ax[i].set_title(img_list[img_id]['slide'])
        
        
def encode_type(df, label_name, label_value):
    """ 
    Args:
        label_name:  name of the label
        label_value: numerical value assigned to the label
    Returns:
        dict of unique label names the appropriate values {label_name: label_value}
    """
    aa = data[[label_name, label_value]].drop_duplicates().sort_values(label_value).reset_index(drop=True)
    return dict(zip(aa[label_name], aa[label_value]))
    

if __name__ == "__main__":
    
    # Path
    datapath = dirpath/'../data'
    
    import ipdb; ipdb.set_trace(context=11)
    
    data = pd.read_csv(datapath/'data_merged.csv')
    csite_enc = encode_type(df=data, label_name='csite', label_value='csite_label')
    ctype_enc = encode_type(df=data, label_name='ctype', label_value='ctype_label')
    CSITE_NUM_CLASSES = len(csite_enc.keys())
    CTYPE_NUM_CLASSES = len(ctype_enc.keys())

