import os
import sys
assert sys.version_info >= (3, 5)

import argparse
from pathlib import Path
from pprint import pprint, pformat
import glob
import shutil
import itertools
import pandas as pd
import numpy as np

from deephistopath.wsi import util

# Utils
from utils.classlogger import Logger
from datasplit.splitter import data_splitter  # cv_splitter
from utils.plots import plot_hist
from utils.utils import dump_dict, get_print_func

# Seed
np.random.seed(42)


fdir = Path(__file__).resolve().parent

MAIN_APPDIR = fdir/'../apps'
DATADIR = fdir/'../data'


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate cv splits.')

    parser.add_argument('-an', '--appname',
                        default=None,
                        type=str,
                        help='App name to dump the splits.')
    parser.add_argument('-ns', '--n_splits',
                        default=1,
                        type=int,
                        help='Number of splits to generate (default: 5).')
    # parser.add_argument('-tem', '--te_method', default='simple', choices=['simple', 'group', 'strat'],
    #                     help='Test split method (default: simple).')
    parser.add_argument('-cvm', '--cv_method',
                        default='simple',
                        choices=['simple', 'group', 'strat'],
                        help='Cross-val split method (default: simple).')
    parser.add_argument('--te_size',
                        default=0.1,
                        help='Test size split (ratio or absolute number) (default: 0.1).')
    # parser.add_argument('--vl_size', type=float, default=0.1, help='Val size split ratio for single split (default: 0.1).')
    parser.add_argument('--split_on',
                        type=str,
                        default=None,
                        choices=['cell', 'drug'],
                        help='Specify which variable (column) to make a hard split on (default: None).')
    parser.add_argument('--ml_task',
                        type=str,
                        default='cls',
                        choices=['reg', 'cls'],
                        help='ML task (default: cls).')
    parser.add_argument('-t', '--trg_name',
                        default='Response',
                        type=str,
                        help='Target column name (required when stratify) (default: rsp).')


    # args = parser.parse_args(args)
    args, other_args = parser.parse_known_args()
    return args


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


def verify_type(s):
    """ Verify that test set type is either int or float. """
    for fn in (float, int):
        try:
            return fn(s)
        except ValueError:
            print('Invalid test size passed.')
    return s


def verify_size(s):
    """ Verify that test set size is valid. """
    s = verify_type(s)
    if isinstance(s, int):
        assert s > 0, f'Test size must be larger than 0. Got {s}.'
    elif isinstance(s, float):
        assert s >= 0 and s <= 1, f'Test set fraction must range between [0, 1]. Got {s}.'
    else:
        raise ValueError(f'Test set fraction must be either int or float. Got {type(s)}.')
    return s


def run(args):

    te_size = verify_size(args.te_size)

    # Path
    appdir = MAIN_APPDIR/args.appname

    # import ipdb; ipdb.set_trace(context=11)

    # Hard split
    split_on = None if args.split_on is None else args.split_on.upper()
    te_method = args.cv_method

    # Specify ML task (regression or classification)
    if args.cv_method == 'strat':
        mltype = 'cls'  # cast mltype to cls in case of stratification
    else:
        mltype = args.ml_task

    # -----------------------------------------------
    #       Create appdir
    # -----------------------------------------------
    gout = appdir/'splits'
    outfigs = gout/'outfigs'
    os.makedirs(gout, exist_ok=True)
    os.makedirs(outfigs, exist_ok=True)

    # -----------------------------------------------
    #       Create logger
    # -----------------------------------------------
    lg = Logger(gout/'data.splitter.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File dir: {fdir}')
    print_fn(f'\n{pformat(vars(args))}')
    dump_dict(vars(args), outpath=gout/'data.splitter.args.txt')  # dump args

    # -----------------------------------------------
    #       Load data
    # -----------------------------------------------
    print_fn('\nLoad master dataset.')
    data = load_data(appdir/'annotations.csv')
    print_fn('data.shape {}'.format(data.shape))

    GE_LEN = sum([1 for c in data.columns if c.startswith('ge_')])
    DD_LEN = sum([1 for c in data.columns if c.startswith('dd_')])

    # import ipdb; ipdb.set_trace(context=11)

    # -----------------------------------------------
    #       Determine the dataset
    # -----------------------------------------------
    ydata = data[args.trg_name] if args.trg_name in data.columns else None
    if (ydata is None) and (args.cv_method == 'strat'):
        raise ValueError('Y data must be specified if splits needs to be stratified.')
    if ydata is not None:
        plot_hist(ydata, title=f'{args.trg_name}', fit=None, bins=100,
                  path=outfigs/f'{args.trg_name}_hist_all.png')

    # -----------------------------------------------
    #       Generate splits (train/val/test)
    # -----------------------------------------------
    print_fn('\n{}'.format('-' * 50))
    print_fn('Split into hold-out train/val/test')
    print_fn('{}'.format('-' * 50))

    kwargs = {'data': data,
              'cv_method': args.cv_method,
              'te_method': te_method,
              'te_size': te_size,
              'mltype': mltype,
              'split_on': split_on
              }

    data_splitter(n_splits=args.n_splits, gout=gout,
                  outfigs=outfigs, ydata=ydata,
                  print_fn=print_fn, **kwargs)

    lg.kill_logger()


def main(args):
    t = util.Time()
    args = parse_args(args)
    run(args)
    t.elapsed_display()
    print('Done.')


if __name__ == "__main__":
    main(sys.argv[1:])
