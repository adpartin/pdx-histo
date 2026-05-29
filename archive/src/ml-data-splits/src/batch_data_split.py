"""
A batch prcoessing code that calls main_data_split.py with the same set of parameters
but different datasets. It generates multiple splits of train/val/test sets and dumps
into respective dirs.

Example:
python src/batch_data_split.py --datadir data/docking_data_march_30 --par_jobs 32 --ml_task cls -t cls --n_splits 100 --gout data/docking_data_march_30
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from glob import glob
from time import time
from joblib import Parallel, delayed

# File path
filepath = Path(__file__).resolve().parent
import main_data_split

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--datadir', required=True, default=None, type=str,
                    help='Data dir where data files are stored (default: None).')
parser.add_argument('--par_jobs', default=1, type=int, 
                    help=f'Number of joblib parallel jobs (default: 1).')
args, other_args = parser.parse_known_args()

# Number of parallel jobs
par_jobs = int( args.par_jobs )
assert par_jobs > 0, f"The arg 'par_jobs' must be at least 1 (got {par_jobs})"

# Data file names
fname_pattern = 'ml.*.parquet'
dfiles = glob( str(Path(args.datadir, fname_pattern)) )

# Main func designed primarily for joblib Parallel
def gen_splits(dfile, *args):
    main_data_split.main([ '--datapath', str(dfile), *args ]) 

# Main execution
t0 = time()
if par_jobs > 1:
    results = Parallel(n_jobs=par_jobs, verbose=1)(
            delayed(gen_splits)(dfile, *other_args) for dfile in dfiles )
else:
    for i, dfile in enumerate(dfiles):
        print('Processing file', dfile)
        gen_splits(dfile, *other_args)
    
t_end = time() - t0
print('Runtime: {:.2f} mins'.format( t_end/60 ))


