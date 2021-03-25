""" Util functions to split data into train/val. """
import pathlib
from pathlib import Path
import numpy as np
from typing import Union, Optional, Any, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from utils.plots import plot_hist


def data_splitter(n_splits: int=1,
                  gout: Optional[Any]=None,
                  outfigs: Optional[Any]=None,
                  ydata=None,
                  print_fn=print,
                  seed: Optional[int]=None,
                  **kwargs) -> Tuple[dict, dict, dict]:
    """
    This func calls get_single_splits() a total of n_splits times to generate
    multiple train/val/test splits.

    Args:
        n_splits : number of splits
        gout : global outdir to dump the splits
        outfigs : outdir to dump the distributions of target variable
        ydata : the target variable (array-like)
        split_on : vol name in the dataframe to use for hard (group) partition
        print_fn : print function
    Return:
        tr_dct, vl_dct, te_dct : tuple of split dicts
    """
    np.random.seed(seed)
    seeds = np.random.choice(n_splits, n_splits, replace=False)

    # These dicts will contain the splits
    tr_dct = {}
    vl_dct = {}
    te_dct = {}

    # import ipdb; ipdb.set_trace()
    for i, seed in enumerate(seeds):
        tr_id, vl_id, te_id = gen_single_split(ydata=ydata, seed=seed, **kwargs)

        tr_dct[i] = tr_id
        vl_dct[i] = vl_id
        te_dct[i] = te_id

        # digits = len(str(n_splits))
        seed_str = str(i)  # f"{seed}".zfill(digits)
        output = '1fold_s' + seed_str

        if gout is not None:
            np.savetxt(gout/f'{output}_tr_id.txt', tr_id.reshape(-1, 1),
                       fmt='%d', delimiter='', newline='\n')
            np.savetxt(gout/f'{output}_vl_id.txt', vl_id.reshape(-1, 1),
                       fmt='%d', delimiter='', newline='\n')
            np.savetxt(gout/f'{output}_te_id.txt', te_id.reshape(-1, 1),
                       fmt='%d', delimiter='', newline='\n')

        if (ydata is not None) and (outfigs is not None):
            plot_hist(ydata[tr_id], title='Train Set Histogram',
                      fit=None, bins=100, path=outfigs/f'{output}_y_hist_train.png')
            plot_hist(ydata[vl_id], title='Val Set Histogram',
                      fit=None, bins=100, path=outfigs/f'{output}_y_hist_val.png')
            plot_hist(ydata[te_id], title='Test Set Histogram',
                      fit=None, bins=100, path=outfigs/f'{output}_y_hist_test.png')
    return (tr_dct, vl_dct, te_dct)


def gen_single_split(data,
                     te_method: str='simple',
                     cv_method: str='simple',
                     te_size: Union[int, float]=0.1,
                     mltask: str='reg',
                     ydata=None,
                     split_on: Optional[str]=None,
                     seed: Optional[int]=None,
                     print_fn=print):
    """
    This func generates train/val/test split indices.

    Args:
        data : dataframe
        te_method : method to split data (D) into test (E) and train (T0)
        cv_method : method to split the test from te_method (T0) into train (T1) and validation (V)
        te_size : fraction of data (D) to extract for test set (E)
        mltask : ml type (task)
        ydata : the target variable
        split_on : vol name in the dataframe to use for hard (group) partition
        seed : see value
        print_fn : print function
    Return:
        tr_id, vl_id, te_id : tuple of 1-D arrays specifying the indices in the original input data
    """
    # import ipdb; ipdb.set_trace()
    np.random.seed(seed)
    idx_vec = np.random.permutation(data.shape[0])
    yvec = ydata.values[idx_vec] if ydata is not None else None

    # Create splitter that splits the full dataset into tr and te
    if te_size < 1:
        te_folds = int(1/te_size)
        test_size = None
    else:
        te_folds = 1
        test_size = te_size

    # TODO: do we really need to specify seed?? idx_vec is already shuffled!
    te_splitter = cv_splitter(cv_method=te_method, cv_folds=te_folds, test_size=test_size,
                              mltask=mltask, shuffle=False, seed=seed)

    te_grp = None if split_on is None else data[split_on].values[idx_vec]
    if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)

    # Split data (D) into tr (T0) and te (E)
    tr_id, te_id = next( te_splitter.split(X=idx_vec, y=yvec, groups=te_grp) )
    tr_id = idx_vec[tr_id] # adjust indices! we'll split the remaining tr into tr and vl
    te_id = idx_vec[te_id] # adjust indices!

    # Update the vector array that excludes the test indices
    idx_vec_ = tr_id; del tr_id
    yvec_ = ydata.values[idx_vec_] if ydata is not None else None

    # Define vl_size while considering the new sample size of the available samples
    if te_size < 1:
        # TODO instead of computing the adjusted ratio for vl_size
        # we can simply use cv_folds=1 and vl_size=te_size!
        vl_size = te_size / (1 - te_size)
        cv_folds = int(1/vl_size)
        test_size = None
    else:
        cv_folds = 1
        test_size = te_size

    # Create splitter that splits tr (T0) into tr (T1) and vl (V)
    cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=test_size,
                     mltask=mltask, shuffle=False, seed=seed)

    cv_grp = None if split_on is None else data[split_on].values[idx_vec_]
    if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)

    # Split tr into tr and vl
    tr_id, vl_id = next( cv.split(X=idx_vec_, y=yvec_, groups=cv_grp) )
    tr_id = idx_vec_[tr_id]  # adjust the indices!
    vl_id = idx_vec_[vl_id]  # adjust the indices!

    # Make sure indices do not overlap
    assert len( set(tr_id).intersection(set(vl_id)) ) == 0, 'Overlapping indices btw tr and vl'
    assert len( set(tr_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and te'
    assert len( set(vl_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and vl'

    # Print split ratios
    print_fn('Train samples {} ({:.2f}%)'.format( len(tr_id), 100*len(tr_id)/data.shape[0] ))
    print_fn('Val   samples {} ({:.2f}%)'.format( len(vl_id), 100*len(vl_id)/data.shape[0] ))
    print_fn('Test  samples {} ({:.2f}%)'.format( len(te_id), 100*len(te_id)/data.shape[0] ))

    # Confirm that group splits are correct (no intersect)
    if split_on is not None:
        print_intersect_on_var(data, tr_id=tr_id, vl_id=vl_id, te_id=te_id,
                               grp_col=split_on, print_fn=print_fn)

    return tr_id, vl_id, te_id


def cv_splitter(cv_method: str='simple',
                cv_folds: int=1,
                test_size: float=0.2,
                mltask: str='reg',
                shuffle: bool=False,
                seed=None):
    """ Creates a cross-validation splitter.

    Args:
        cv_method: 'simple', 'stratify' (only for classification), 'groups' (only for regression)
        cv_folds: number of cv folds
        test_size: fraction of test set size (used only if cv_folds=1)
        mltask: 'reg', 'cls'
    """
    # Classification
    if mltask == 'cls':
        if cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size,
                                  random_state=seed)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=shuffle,
                           random_state=seed)

        elif cv_method == 'strat':
            if cv_folds == 1:
                cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=test_size,
                                            random_state=seed)
            else:
                # cv = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle,
                #                      random_state=seed)
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=False,
                                     random_state=None)

        elif cv_method == 'group':
            if cv_folds == 1:
                raise ValueError('GroupShuffleSplit splits groups based on group \
                                 count and not based on sample count.')
                # https://github.com/scikit-learn/scikit-learn/issues/13369
                # https://github.com/scikit-learn/scikit-learn/issues/9193
                cv = GroupShuffleSplit(n_splits=cv_folds, test_size=test_size,
                                       random_state=seed)
            else:
                cv = GroupKFold(n_splits=cv_folds)

    # Regression
    elif mltask == 'reg':
        if cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size,
                                  random_state=seed)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=shuffle,
                           random_state=seed)

        elif cv_method == 'group':
            if cv_folds == 1:
                raise ValueError('GroupShuffleSplit splits groups based on group \
                                 count and not based on sample count.')
                # https://github.com/scikit-learn/scikit-learn/issues/13369
                # https://github.com/scikit-learn/scikit-learn/issues/9193
                cv = GroupShuffleSplit(n_splits=cv_folds, test_size=test_size,
                                       random_state=seed)
            else:
                cv = GroupKFold(n_splits=cv_folds)
    return cv


def print_intersect_on_var(df, tr_id, vl_id, te_id, grp_col='CELL', print_fn=print):
    """ Print intersection between train, val, and test datasets with respect
    to grp_col column if provided. df is usually a metadata.
    """
    if grp_col in df.columns:
        tr_grp_unq = set( df.loc[tr_id, grp_col] )
        vl_grp_unq = set( df.loc[vl_id, grp_col] )
        te_grp_unq = set( df.loc[te_id, grp_col] )
        print_fn(f'\tTotal intersects on {grp_col} btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}')
        print_fn(f'\tTotal intersects on {grp_col} btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}')
        print_fn(f'\tTotal intersects on {grp_col} btw vl and te: {len(vl_grp_unq.intersection(te_grp_unq))}')
        print_fn(f'\tUnique {grp_col} in tr: {len(tr_grp_unq)}')
        print_fn(f'\tUnique {grp_col} in vl: {len(vl_grp_unq)}')
        print_fn(f'\tUnique {grp_col} in te: {len(te_grp_unq)}')
    else:
        raise(f'The column {grp_col} was not found!')


def split_size(x):
    """ Split size can be float (0, 1) or int (casts value as needed). """
    assert x > 0, 'Split size must be greater than 0.'
    return int(x) if x > 1.0 else x


def plot_ytr_yvl_dist(ytr, yvl, title=None, outpath='.'):
    """ Plot distributions of response data of train and val sets. """
    fig, ax = plt.subplots()
    plt.hist(ytr, bins=100, label='ytr', color='b', alpha=0.5)
    plt.hist(yvl, bins=100, label='yvl', color='r', alpha=0.5)
    if title is None: title = ''
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    if outpath is None:
        plt.savefig(Path(outpath)/'ytr_yvl_dist.png', bbox_inches='tight')
    else:
        plt.savefig(outpath, bbox_inches='tight')
