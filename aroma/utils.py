"""
Miscellaneous utility functions
"""
import os.path as op

import numpy as np


def motpars_spm2fsl(motpars):
    """
    Convert SPM format motion parameters to FSL format
    """
    if isinstance(motpars, str) and op.isfile(motpars):
        motpars = np.loadtxt(motpars)

    # Split translations from rotations
    trans, rot = motpars[:, :3], motpars[:, 3:]

    # Convert rotations in degrees to radians
    rot *= np.pi / 180.

    # Place rotations first
    motpars_fsl = np.hstack((rot, trans))
    return motpars_fsl


def motpars_afni2fsl(motpars):
    """
    Convert AFNI format motion parameters to FSL format
    """
    if isinstance(motpars, str) and op.isfile(motpars):
        motpars = np.loadtxt(motpars)

    # Split translations from rotations
    trans, rot = motpars[:, :3], motpars[:, 3:]

    # Convert rotations in degrees to radians
    rot *= np.pi / 180.

    # Place rotations first
    motpars_fsl = np.hstack((rot, trans))
    return motpars_fsl


def determine_motpar_source(file_):
    if file_.startswith('rp_') and file_.endswith('.txt'):
        return 'spm'
    elif file_.endswith('.1D'):
        return 'afni'
    elif file_.endswith('.txt'):
        return 'fsl'


def cross_correlation(a, b):
    """Cross Correlations between columns of two matrices"""
    assert a.ndim == b.ndim == 2
    _, ncols_a = a.shape
    # nb variables in columns rather than rows hence transpose
    # extract just the cross terms between cols in a and cols in b
    return np.corrcoef(a.T, b.T)[:ncols_a, ncols_a:]


def get_resource_path():
    """
    Returns the path to general resources, terminated with separator. Resources
    are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), 'resources') + op.sep)
