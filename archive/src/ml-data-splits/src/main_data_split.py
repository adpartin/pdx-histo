"""
This code generates multiple splits of train/val/test sets.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pprint, pformat
from typing import Union

import numpy as np

# Utils
from utils.classlogger import Logger
from datasplit.splitter import data_splitter  # cv_splitter
from utils.plots import plot_hist
from utils.utils import load_data, dump_dict, get_print_func

# File path
fdir = Path(__file__).resolve().parent

# Seed
seed = 42
# np.random.seed(seed)


def parse_args(args):
    parser = argparse.ArgumentParser(description='Create train/val/test splits.')

    parser.add_argument('-dp', '--datapath',
                        required=True,
                        default=None,
                        type=str,
                        help='Full path to the data (default: None).')
    parser.add_argument('--gout',
                        default=None,
                        type=str,
                        help='Global outdir to dump the splits.')
    parser.add_argument('-ns', '--n_splits',
                        default=5,
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
                        choices=['Sample', 'slide', 'Group'],
                        help='Specify which variable (column) to make a hard split on (default: None).')
    parser.add_argument('--ml_task',
                        type=str,
                        default='reg',
                        choices=['reg', 'cls'],
                        help='ML task (default: reg).')
    parser.add_argument('-t', '--trg_name',
                        default=None,
                        type=str,
                        help='Target column name (required when stratify) (default: None).')

    args = parser.parse_args(args)
    return args


def verify_size(s: Union[int, float]) -> Union[int, float]:
    """ Verify that te_size is either int or float. """
    from ast import literal_eval
    try:
        s = literal_eval(s)
        if isinstance(s, int) or isinstance(s, float):
            return s
    except ValueError:
        print('te_size must be either int or float.')
        sys.exit()

    # for fn in (int, float):
    #     try:
    #         return fn(s)
    #     except ValueError:
    #         print('te_size must be either int or float.')
    #         sys.exit()


def run(args):

    print("\nInput args:")
    pprint(vars(args))

    t0 = time()
    te_size = verify_size(args.te_size)
    datapath = Path(args.datapath).resolve()

    # Hard split
    # split_on = None if args.split_on is None else args.split_on.upper()
    cv_method = args.cv_method
    te_method = cv_method

    # Specify ML task (regression or classification)
    if cv_method == "strat":
        mltask = "cls"  # cast mltask to cls in case of stratification
    else:
        mltask = args.ml_task

    # Target column name
    trg_name = str(args.trg_name)
    # assert args.trg_name in data.columns, f'The prediction target ({args.name}) \
    #     was not found in the dataset.'

    # import ipdb; ipdb.set_trace()

    # -----------------------------------------------
    #       Create outdir
    # -----------------------------------------------
    if args.gout is not None:
        gout = Path(args.gout).resolve()
        sufx = "none" if args.split_on is None else args.split_on
        gout = gout/datapath.with_suffix(".splits")
        if args.split_on is not None:
            gout = gout/f"split_on_{sufx}"
        else:
            gout = gout/f"split_on_none"
    else:
        # Note! useful for drug response
        sufx = "none" if args.split_on is None else args.split_on
        gout = datapath.with_suffix(".splits")

    outfigs = gout/"outfigs"
    os.makedirs(gout, exist_ok=True)
    os.makedirs(outfigs, exist_ok=True)

    # -----------------------------------------------
    #       Create logger
    # -----------------------------------------------
    lg = Logger(gout/"data.splitter.log")
    print_fn = get_print_func(lg.logger)
    print_fn(f"File path: {fdir}")
    print_fn(f"\n{pformat(vars(args))}")
    dump_dict(vars(args), outpath=gout/"data.splitter.args.txt")

    # -----------------------------------------------
    #       Load data
    # -----------------------------------------------
    print_fn("\nLoad master dataset.")
    data = load_data(datapath)
    print_fn("data.shape {}".format(data.shape))

    # ydata = data[trg_name] if trg_name in data.columns else None
    # if (cv_method == "strat") and (ydata is None):
    #     raise ValueError("Prediction target column must be available if splits need to be stratified.")

    if (cv_method == "strat") and (trg_name not in data.columns):
        raise ValueError("Prediction target column must be available if splits need to be stratified.")

    # if ydata is not None:
    #     plot_hist(ydata, title=f"{trg_name}", fit=None, bins=100,
    #               path=outfigs/f"{trg_name}_hist_all.png")

    if trg_name in data.columns:
        plot_hist(data[trg_name], title=f"{trg_name}", fit=None, bins=100,
                  path=outfigs/f"{trg_name}_hist_all.png")

    # -----------------------------------------------
    #       Generate splits (train/val/test)
    # -----------------------------------------------
    print_fn("\n{}".format("-" * 50))
    print_fn("Split data into hold-out train/val/test")
    print_fn("{}".format("-" * 50))

    kwargs = {"cv_method": cv_method,
              "te_method": te_method,
              "te_size": te_size,
              "mltask": mltask,
              "split_on": args.split_on
              }

    data_splitter(data = data,
                  n_splits = args.n_splits,
                  gout = gout,
                  outfigs = outfigs,
                  # ydata = ydata,
                  target_name = trg_name,
                  print_fn = print_fn,
                  seed = seed,
                  **kwargs)

    print_fn("Runtime: {:.1f} min".format( (time()-t0)/60) )
    print_fn("Done.")
    lg.close_logger()


def main(args):
    args = parse_args(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
