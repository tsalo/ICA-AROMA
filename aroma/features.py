#!/usr/bin/env python
"""
Functions for ICA-AROMA v0.3 beta
"""
from __future__ import division
from __future__ import print_function

import os
import os.path as op
import random
import subprocess

import numpy as np

from .utils import correlate_columns, get_resource_path


def feature_time_series(mix, rp6):
    """
    This function extracts the maximum RP correlation feature scores.
    It determines the maximum robust correlation of each component time-series
    with a model of 72 realignment parameters.

    Parameters
    ---------------------------------------------------------------------------------
    melmix:     Full path of the melodic_mix text file
    mc:     Full path of the text file containing the realignment parameters

    Returns
    ---------------------------------------------------------------------------------
    maxRPcorr:  Array of the maximum RP correlation feature scores for the components
    of the melodic_mix file
    """
    assert rp6.shape[0] == mix.shape[0]

    # Determine the derivatives of the RPs (add zeros at time-point zero)
    _, nparams = rp6.shape
    rp6_der = np.vstack((np.zeros(nparams), np.diff(rp6, axis=0)))

    # Create an RP-model including the RPs and its derivatives
    rp12 = np.hstack((rp6, rp6_der))

    # add the squared RP-terms to the model
    rp24 = np.hstack((rp12, rp12 ** 2))

    # add the fw and bw shifted versions
    # update to go through +- 5
    # NOTE: This is for fast cross-correlation
    rp_model = rp24.copy()
    N_SHIFT = 5
    for i_shift in range(1, N_SHIFT+1):
        rp24_fw = np.vstack((np.zeros(i_shift, rp24.shape[1]), rp24[:-i_shift]))
        rp24_bw = np.vstack((rp24[i_shift:], np.zeros(i_shift, rp24.shape[1])))
        rp_model = np.hstack((rp_model, rp24_fw, rp24_bw))

    # Determine the maximum correlation between RPs and IC time-series
    N_SPLITS = 1000
    n_vols, n_components = mix.shape
    nrows_to_choose = int(round(0.9 * n_vols))

    # Max correlations for multiple splits of the dataset (for a robust estimate)
    max_corrs = np.empty((N_SPLITS, n_components))
    for i_split in range(N_SPLITS):
        # Select a random subset of 90% of the dataset rows (*without* replacement)
        chosen_rows = random.sample(population=range(n_vols),
                                    k=nrows_to_choose)

        # Combined correlations between RP and IC time-series, squared and non squared
        corrs = correlate_columns(mix[chosen_rows], rp_model[chosen_rows])

        # Maximum absolute temporal correlation for every IC
        max_corrs[i_split] = np.abs(corrs).max(axis=1)

    # Feature score is the mean of the maximum correlation over all the random splits
    # Avoid propagating occasional nans that arise in artificial test cases
    max_rp_corr = np.nanmean(max_corrs, axis=0)
    return max_rp_corr


def feature_frequency(ft_data, freqs, TR):
    """
    This function extracts the high-frequency content feature scores.
    It determines the frequency, as fraction of the Nyquist frequency,
    at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    Parameters
    ---------------------------------------------------------------------------------
    ft_data : (F x C) :obj:`numpy.ndarray`
        Fourier-transformed ICA component time series. Component by frequency array.
    freqs : (F,) :obj:`numpy.ndarray`
        Frequencies in Hz.
    TR : :obj:`float`
        TR in seconds of data

    Returns
    ---------------------------------------------------------------------------------
    HFC : (C,) :obj:`numpy.ndarray`
        Array of the HFC ('high-frequency content') feature scores for the
        components of the ft_data array
    """
    assert ft_data.shape[0] == freqs.shape[0]

    # Determine sample frequency
    Fs = 1 / TR

    # Determine Nyquist-frequency
    Ny = Fs / 2

    # Only include frequencies higher than 0.01Hz
    fincl = np.squeeze(np.array(np.where(freqs > 0.01)))
    ft_data = ft_data[fincl, :]
    freqs = freqs[fincl]

    # Set frequency range to [0-1]
    f_norm = (freqs - 0.01) / (Ny - 0.01)

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fract = np.cumsum(ft_data, axis=0) / np.sum(ft_data, axis=0)

    # Determine the index of the frequency with the fractional cumulative sum closest to 0.5
    idx_cutoff = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)

    # Now get the fractions associated with those indices index, these are the final feature scores
    HFC = f_norm[idx_cutoff]

    # Return feature score
    return HFC


def feature_spatial(z_thresh_maps, csf_mask="auto", out_mask="auto",
                    edge_mask="auto"):
    """
    This function extracts the spatial feature scores. For each IC it
    determines the fraction of the mixture modeled thresholded Z-maps
    respectively located within the CSF or at the brain edges, using predefined
    standardized masks.

    Parameters
    ----------
    z_thresh_maps : (X x Y x Z x C) :obj:`numpy.ndarray`
        Array containing mixture-modeled thresholded (p>0.5) z-maps
    csf_mask, out_mask, edge_mask : (X x Y x Z) :obj:`numpy.ndarray`
        Masks of CSF, nonbrain, and brain edges

    Returns
    -------
    edgeFract : (C,) :obj:`numpy.ndarray`
        Array of the edge fraction feature scores for the components of the
        melIC file
    csfFract : (C,) :obj:`numpy.ndarray`
        Array of the CSF fraction feature scores for the components of the
        melIC file
    """
    n_components = z_thresh_maps.shape[-1]

    # Get absolute values
    z_thresh_maps = np.abs(z_thresh_maps)

    if csf_mask == "auto":
        csf_mask = op.join(get_resource_path(), "mask_csf.nii.gz")
    else:
        assert op.isfile(csf_mask)

    if out_mask == "auto":
        out_mask = op.join(get_resource_path(), "mask_out.nii.gz")
    else:
        assert op.isfile(out_mask)

    if edge_mask == "auto":
        edge_mask = op.join(get_resource_path(), "mask_edge.nii.gz")
    else:
        assert op.isfile(edge_mask)

    # Loop over ICs
    edge_fract = np.zeros(n_components)
    csf_fract = np.zeros(n_components)
    for i_comp in range(n_components):
        comp_map = z_thresh_maps[..., i_comp]
        z_total_sum = np.sum(comp_map)
        if z_total_sum == 0:
            LGR.warning('The spatial map of component {0} is empty. Please '
                        'check!'.format(i_comp))

        csf_data = comp_map[csf_mask]
        z_csf_sum = np.sum(csf_data)

        edge_data = comp_map[edge_mask]
        z_edge_sum = np.sum(edge_data)

        out_data = comp_map[out_mask]
        z_out_sum = np.sum(out_data)

        # Determine edge and CSF fraction
        if z_total_sum != 0:
            edge_fract[i_comp] = (z_out_sum + z_edge_sum) / (z_total_sum - z_csf_sum)
            csf_fract[i_comp] = z_csf_sum / z_total_sum
        else:
            edge_fract[i_comp] = 0.
            csf_fract[i_comp] = 0.

    return edge_fract, csf_fract
