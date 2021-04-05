from pathlib import Path
from typing import List
import os
import glob
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    if int(tf.__version__.split('.')[0]) < 2:
        import keras
        from keras.models import load_model
        from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
        from keras.utils import plot_model
    else:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
        from tensorflow.keras.utils import plot_model

        from tensorflow.keras import backend as K
        from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
        from tensorflow.keras import optimizers
        from tensorflow.keras.optimizers import SGD, Adam
        from tensorflow.keras.models import Sequential, Model
except:
    print('Could not import tensorflow.')
    

def clr_keras_callback(mode=None, base_lr=1e-4, max_lr=1e-3, gamma=0.999994):
    """ Creates keras callback for cyclical learning rate. """
    # keras_contrib = './keras_contrib/callbacks'
    # sys.path.append(keras_contrib)
    from cyclical_learning_rate import CyclicLR
    if mode == 'trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif mode == 'trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif mode == 'exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range', gamma=gamma) # 0.99994; 0.99999994; 0.999994
    return clr


def r2_krs(y_true, y_pred):
    # from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def save_krs_history(history, outdir='.'):
    fname = 'krs_history.csv'
    hh = pd.DataFrame( history.history )
    hh['epoch'] = np.asarray(history.epoch) + 1    
    hh.to_csv( Path(outdir)/fname, index=False )
    return hh


def capitalize_metric(met):
    return ' '.join(s.capitalize() for s in met.split('_'))


def plot_prfrm_metrics(history=None, logfile_path=None, title=None, name=None, skp_ep=0, outdir='.', add_lr=False):    
    """ Plots training curves from keras history or keras traininig.log file.
    Args:
        history : history variable of keras
        path_to_logs : full path to the training log file of keras
        skp_ep : number of epochs to skip when plotting metrics 
        add_lr : add curve of learning rate progression over epochs
        
    Retruns:
        history : keras history
    """
    
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if history is not None:
        # Plot from keras history
        hh = history.history
        epochs = np.asarray(history.epoch) + 1
        all_metrics = list(history.history.keys())
        
    elif logfile_path is not None:
        # Plot from keras training.log file
        hh = pd.read_csv(logfile_path, sep=',', header=0)
        epochs = hh['epoch'] + 1
        all_metrics = list(hh.columns)
        
    # Get training performance metrics
    pr_metrics = ['_'.join(m.split('_')[1:]) for m in all_metrics if 'val' in m]

    if len(epochs) <= skp_ep: skp_ep = 0
    eps = epochs[skp_ep:]
        
    # Interate over performance metrics and plot
    for p, m in enumerate(pr_metrics):
        metric_name = m
        metric_name_val = 'val_' + m

        y_tr = hh[metric_name][skp_ep:]
        y_vl = hh[metric_name_val][skp_ep:]
        
        ymin = min(set(y_tr).union(y_vl))
        ymax = max(set(y_tr).union(y_vl))
        lim = (ymax - ymin) * 0.1
        ymin, ymax = ymin - lim, ymax + lim

        # Start figure
        fig, ax1 = plt.subplots()
        
        # Plot metrics
        alpha = 0.6
        linewidth = 1
        fontsize = 12
        ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=linewidth,
                 alpha=alpha, label=capitalize_metric(metric_name))
        ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=linewidth,
                 alpha=alpha, label=capitalize_metric(metric_name_val))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(capitalize_metric(metric_name))
        ax1.set_xlim([min(eps)-1, max(eps)+1])
        ax1.set_ylim([ymin, ymax])
        ax1.tick_params('y', colors='k')
        
        # ax1.tick_params(axis='both', which='major', labelsize=12)
        # ax1.tick_params(axis='both', which='minor', labelsize=12)        
        
        # Add learning rate
        if logfile_path is None: # learning rate is not logged into log file (so it's only available from history)
            if (add_lr is True) and ('lr' in hh):
                ax2 = ax1.twinx()
                ax2.plot(eps, hh['lr'][skp_ep:], color='g', marker='.', linestyle=':',
                         linewidth=linewidth, alpha=alpha, markersize=5, label='LR')
                ax2.set_ylabel('Learning rate', color='g', fontsize=fontsize)

                ax2.set_yscale('log') # 'linear'
                ax2.tick_params('y', colors='g')
        
        ax1.grid(True)
        # plt.legend([metric_name, metric_name_val], loc='best')
        # medium.com/@samchaaa/how-to-plot-two-different-scales-on-one-plot-in-matplotlib-with-legend-46554ba5915a
        legend = ax1.legend(loc='best', prop={'size': 10})
        frame = legend.get_frame()
        frame.set_facecolor('0.95')
        if title is not None: plt.title(title)
        
        # fig.tight_layout()
        fname = (metric_name + '.png') if name is None else (name + '_' + metric_name + '.png')
        figpath = Path(outdir) / fname
        plt.savefig(figpath, bbox_inches='tight')
        plt.close()
        
    return history


def calc_records_in_tfr_folder(tfr_dir):
    """
    Calc and print the number of examples (tiles) in all tfrecords in the
    input folder.
    """
    count = 0
    for tfr_path in sorted(tfr_dir.glob('*.tfrec*')):
        count += sum(1 for _ in tf.data.TFRecordDataset(str(tfr_path)))
    print('Number of examples in all tfrecords in the folder:', count)

    
def calc_records_in_tfr_files(tfr_files):
    """
    Calc and print the number of examples (tiles) in all tfrecords in the
    provided list of file paths.
    """
    count = 0
    for tfr_path in sorted(tfr_files):
        count += sum(1 for _ in tf.data.TFRecordDataset(str(tfr_path)))
    print('Number of examples in all tfrecords in the folder:', count)
    return count
    
    
def count_data_items(filenames):
    import re
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
    
    
def calc_examples_in_tfrecord(tfr_path):
    """
    Calc and print the number of examples (tiles) in the input tfrecord
    file provided by the path to the file.
    """
    count = sum(1 for _ in tf.data.TFRecordDataset(str(tfr_path)))
    print('Number of examples in the tfrecord:', count)


def get_tfr_files(tfr_dir, tfr_names) -> List[str]:
    """ Get filenames of tfrecords based on response sample names.

    Args:
        fnames : file names exluding the suffix '.tfrecord*'

    Example:
        get_tfr_files(tr_smp_names)
    """
    # tfr_all_files = sorted(tfr_dir.glob('*.tfrec*'))
    tfr_all_files = sorted(glob.glob(f"{str(tfr_dir)}/*.tfrec*"))
    tfr_sub_files = []

    # suffix = np.unique([s.suffix for s in tfr_all_files])
    suffix = np.unique([os.path.splitext(s)[-1] for s in tfr_all_files])
    if len(suffix) < 1:
        raise FileNotFoundError("No files found with suffix '.tfrec*'") 
    if len(suffix) > 1:
        "Multiple files with variations of '.tfrec*' suffix found."
    suffix = suffix[0]

    tfr_query_files = [str(tfr_dir/(fname + suffix)) for fname in tfr_names]
    tfr_sub_files = list(set(tfr_query_files).intersection(set(tfr_all_files)))

    # # Takes too long!
    # for ii, sname in enumerate(tfr_names):
    #     fname = tfr_dir/(sname + suffix)
    #     if fname not in tfr_all_files:
    #         raise ValueError(f"File was not found in:\n\t{fname}")
    #     tfr_sub_files.append(str(fname))

    return tfr_sub_files


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
