import pandas as pd
import numpy as np
from tqdm import tqdm
import inspect

from muon_regression.scripts.prep_data import singleroot_to_hdf5


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
    # TODO: what if calo shape is (1,1,50,32,32)? We remain with three dims
    if len(data.shape) == 5:
        return data.squeeze()
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
    return calorimeter.sum(axis=(1, 2, 3), where=(calorimeter > threshold))


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
    return calorimeter.sum(axis=(1, 2, 3), where=(calorimeter <= threshold))


def muon_features(features: list, use_files, limit_files=None, write=False, output_file=None, **kwargs):
    """Writes a csv of shape (true energy, *features), where each row corresponds to a muon.

    Parameters
    ----------
    features : list of Callable objects
        [description]
    use_files : str
        Path to .txt file containing .root files to process
    limit_files : [type], optional
        [description], by default None
    write : bool, optional
        [description], by default False
    output_file : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    with open(use_files, 'r') as fileinput:
        # -1 is needed because file is written with \n at the end
        files = list(sorted(fileinput.read().split('\n')[:-1]))

    if limit_files is None:
        limit_files = len(files)

    dfs = []
    for file in tqdm(files[:limit_files], desc='Extracting features from muons in each file'):
        muon_data = singleroot_to_hdf5(in_file=file, write=False)
        calorimeters = muon_data['e'][()]  # [()] "gets" the array from the hdf5 dataset
        true_energies = muon_data['y'][()]
        file_df = pd.DataFrame({'true_energy': true_energies})
        for feature in features:
            # get only feature-specific kwargs
            feature_kwargs = {key: kwargs[key] for key in (kwargs.keys() & inspect.getfullargspec(feature)[0])}
            file_df[feature.__name__] = feature(calorimeter=calorimeters, **feature_kwargs)
        dfs.append(file_df)

    out_df = pd.concat(dfs, axis=0, ignore_index=True)
    if write:
        out_df.to_csv(output_file, index=False)

    return out_df
