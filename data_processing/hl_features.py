import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm

import sys
sys.path.append('../..')

from muon_regression.scripts import singleroot_to_hdf5

def squeeze_data(data):
    """If data has an extra dimension due to ROOT conversion in muon_regression code, squeeze it!

    Parameters
    ----------
    data : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    if len(data.shape) == 5:
        return data = data.squeeze()
    elif len(data.shape) != 4:
        raise ValueError(f'Expected data shape is 4 or 5, got {data.shape}')
    else:
        return data

def v0(calorimeter, threshold):
    """Compute high-level feature V[0].
    See Dorigo et al., August 2020, section 3.2.

    Parameters
    ----------
    calorimeter : np.ndarray
        Expected shape: (n_muons, 1, z-dim, x-dim, y-dim) or (n_muons, z-dim, x-dim, y-dim)
    threshold : float
        [description]

    Returns
    -------
    float
        [description]
    """
    calorimeter = squeeze_data(calorimeter)
    return np.sum(calorimeter.sum(axis=(1,2,3)), where=(calorimeter > threshold))


def v1(calorimeter, threshold):
    """Compute high-level feature V[1].
    See Dorigo et al., August 2020, section 3.2.

    Parameters
    ----------
    calorimeter : np.ndarray
        Expected shape: (n_muons, 1, z-dim, x-dim, y-dim) or (n_muons, z-dim, x-dim, y-dim)
    threshold : float
        [description]

    Returns
    -------
    float
        [description]
    """
    calorimeter = squeeze_data(calorimeter)
    return np.sum(calorimeter.sum(axis=(1,2,3)), where=(calorimeter <= threshold))


def muon_features(muons, features: list, use_files, write=False, output_file=None):
    """Writes a csv of shape (true energy, *features), where each row corresponds to a muon.

    Parameters
    ----------
    muons : np.ndarray
        Expected shape: n_muons, 1, z-dim, x-dim, y-dim 
    features : list-like of Callable objects
        [description]
    use_files : str
        Path to .txt file containing .root files to process
    write : bool, optional
        [description], by default False
    output_file : [type], optional
        [description], by default None
    """

    with open(use_files) as fileinput:
        # -1 is needed because file is written with \n at the end
        files = list(sorted(fileinput.read().split('\n')[:-1]))
    
    list_df = []
    for file in tqdm(files, desc='Extracting features from muons in each file'):
        muon_data = singleroot_to_hdf5(in_file=pathlib.Path(file), write=True)
        for feature in features: 
            calorimeters = muon_data['E'][()]  # [()] "gets" the array from the hdf5 dataset
            true_energies = muon_data['y'][()]
            # TODO: passing features in this way makes dispatching parameters an unnecessary difficult task



