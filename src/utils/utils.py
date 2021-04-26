import json
import os
import pandas as pd
from pathlib import Path
from pprint import pprint, pformat
import sys
from time import time
# from sklearn.preprocessing import LabelEncoder


def verify_path(path):
    """ Verify that the path exists. """
    if path is None:
        sys.exit('Program terminated. You must specify a correct path.')
    path = Path(path)
    assert path.exists(), f'The specified path was not found: {path}.'
    return path


def load_data( datapath, file_format=None ):
    datapath = verify_path( datapath )
    if file_format is None:
        file_format = str(datapath).split('.')[-1]

    if file_format == 'parquet':
        data = pd.read_parquet( datapath )
    elif file_format == 'hdf5':
        data = pd.read_hdf5( datapath )
    elif file_format == 'csv':
        data = pd.read_csv( datapath )
    else:
        try:
            data = pd.read_csv( datapath )
        except:
            print('Cannot load file', datapath)
    return data


def drop_dup_rows(data, print_fn=print):
    """ Drop duplicate rows. """
    print_fn('\nDrop duplicates ...')
    cnt0 = data.shape[0]; print_fn('Samples: {}'.format( cnt0 ))
    data = data.drop_duplicates().reset_index(drop=True)
    cnt1 = data.shape[0]; print_fn('Samples: {}'.format( cnt1 ));
    print_fn('Dropped duplicates: {}'.format( cnt0-cnt1 ))
    return data


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open(Path(outpath), 'w') as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))


def get_print_func(logger=None):
    """ Returns the python 'print' function if logger is None. Othersiwe, returns logger.info. """
    return print if logger is None else logger.info


def create_outdir(outdir, args=None):
    from datetime import datetime
    t = datetime.now()
    t = [t.year, "-", str(t.month).zfill(2), "-", str(t.day).zfill(2),
         "_", "h", str(t.hour).zfill(2), "-", "m", str(t.minute).zfill(2)]
    t = "".join([str(i) for i in t])

    outdir = Path(outdir)
    outdir = outdir/t
    
    os.makedirs(outdir)
    return outdir    


def fea_types_to_str_name(params, sep="_"):
    """ Extract the specified feature types from params and create a str that
    specifies which features are used. """
    fea_types_str = []
    # keys = ["ge", "tile"]

    if params.use_tile is True:
        fea_types_str.append("tile")

    if params.use_ge is True:
        fea_types_str.append("ge")

    if params.use_dd1 is True:
        fea_types_str.append("dd1")

    if params.use_dd2 is True:
        fea_types_str.append("dd2")

    fea_types_str = f"{sep}".join([str(i) for i in fea_types_str])
    return fea_types_str


def create_outdir_2(outdir, params=None):
    from datetime import datetime
    t = datetime.now()
    t = [t.year, "-", str(t.month).zfill(2), "-", str(t.day).zfill(2),
         "_", "h", str(t.hour).zfill(2), "-", "m", str(t.minute).zfill(2)]
    t = "".join([str(i) for i in t])

    # inputs = []
    # # keys = ["ge", "tile"]

    # if params.use_tile is True:
    #     inputs.append("tile")

    # if params.use_ge is True:
    #     inputs.append("ge")

    # if params.use_dd1 is True:
    #     inputs.append("dd1")

    # if params.use_dd2 is True:
    #     inputs.append("dd2")

    # inputs = "_".join([str(i) for i in inputs])
    name = fea_types_to_str_name(params) + "_" + t

    outdir = Path(outdir)
    outdir = outdir/name
    
    os.makedirs(outdir)
    return outdir    


def read_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return lines


def cast_list(ll, dtype=int):
    return [dtype(i) for i in ll]


class Params():
    """
    Taken from:
    github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/vision/model/utils.py

    Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


class Timer:
  """
  Measure runtime.
  """
  def __init__(self):
    self.start = time()

  def timer_end(self):
    self.end = time()
    time_diff = self.end - self.start
    return time_diff

  def display_timer(self, print_fn=print):
    time_diff = self.timer_end()
    if (time_diff)//3600 > 0:
        print_fn("Runtime: {:.1f} hrs".format( (time_diff)/3600) )
    else:
        print_fn("Runtime: {:.1f} mins".format( (time_diff)/60) )
