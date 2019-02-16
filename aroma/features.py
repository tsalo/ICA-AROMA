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

from .utils import cross_correlation, get_resource_path


def feature_time_series(melmix, mc):
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
    # Read melodic mix file (IC time-series), subsequently define a set of squared time-series
    if op.isfile(melmix):
        mix = np.loadtxt(melmix)
    elif isinstance(melmix, np.ndarray):
        mix = melmix
    else:
        raise ValueError("Is bad")

    # Read motion parameter file
    if op.isfile(mc):
        rp6 = np.loadtxt(mc)
    elif isinstance(mc, np.ndarray):
        rp6 = mc
    else:
        raise ValueError("Is bad")
    assert rp6.shape[0] == mix.shape[0]

    # Determine the derivatives of the RPs (add zeros at time-point zero)
    _, nparams = rp6.shape
    rp6_der = np.vstack((np.zeros(nparams), np.diff(rp6, axis=0)))

    # Create an RP-model including the RPs and its derivatives
    rp12 = np.hstack((rp6, rp6_der))

    # Add the squared RP-terms to the model
    # add the fw and bw shifted versions
    rp12_1fw = np.vstack((np.zeros(2 * nparams), rp12[:-1]))
    rp12_1bw = np.vstack((rp12[1:], np.zeros(2 * nparams)))
    rp_model = np.hstack((rp12, rp12_1fw, rp12_1bw))

    # Determine the maximum correlation between RPs and IC time-series
    nsplits = 1000
    nmixrows, nmixcols = mix.shape
    nrows_to_choose = int(round(0.9 * nmixrows))

    # Max correlations for multiple splits of the dataset (for a robust estimate)
    max_correls = np.empty((nsplits, nmixcols))
    for i in range(nsplits):
        # Select a random subset of 90% of the dataset rows (*without* replacement)
        chosen_rows = random.sample(population=range(nmixrows), k=nrows_to_choose)

        # Combined correlations between RP and IC time-series, squared and non squared
        correl_nonsquared = cross_correlation(mix[chosen_rows], rp_model[chosen_rows])
        correl_squared = cross_correlation(
            mix[chosen_rows] ** 2, rp_model[chosen_rows] ** 2
        )
        correl_both = np.hstack((correl_squared, correl_nonsquared))

        # Maximum absolute temporal correlation for every IC
        max_correls[i] = np.abs(correl_both).max(axis=1)

    # Feature score is the mean of the maximum correlation over all the random splits
    # Avoid propagating occasional nans that arise in artificial test cases
    return np.nanmean(max_correls, axis=0)


def feature_frequency(melFTmix, TR):
    """
    This function extracts the high-frequency content feature scores.
    It determines the frequency, as fraction of the Nyquist frequency,
    at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    Parameters
    ---------------------------------------------------------------------------------
    melFTmix:   Full path of the melodic_FTmix text file
    TR:     TR (in seconds) of the fMRI data (float)

    Returns
    ---------------------------------------------------------------------------------
    HFC:        Array of the HFC ('High-frequency content') feature scores
    for the components of the melodic_FTmix file
    """
    # Determine sample frequency
    Fs = 1 / TR

    # Determine Nyquist-frequency
    Ny = Fs / 2

    # Load melodic_FTmix file
    if op.isfile(melFTmix):
        FT = np.loadtxt(melFTmix)
    elif isinstance(melFTmix, np.ndarray):
        FT = melFTmix
    else:
        raise ValueError("Is bad")

    # Determine which frequencies are associated with every row in the
    # melodic_FTmix file  (assuming the rows range from 0Hz to Nyquist)
    f = Ny * (np.array(list(range(1, FT.shape[0] + 1)))) / (FT.shape[0])

    # Only include frequencies higher than 0.01Hz
    fincl = np.squeeze(np.array(np.where(f > 0.01)))
    FT = FT[fincl, :]
    f = f[fincl]

    # Set frequency range to [0-1]
    f_norm = (f - 0.01) / (Ny - 0.01)

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fract = np.cumsum(FT, axis=0) / np.sum(FT, axis=0)

    # Determine the index of the frequency with the fractional cumulative sum closest to 0.5
    idx_cutoff = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)

    # Now get the fractions associated with those indices index, these are the final feature scores
    HFC = f_norm[idx_cutoff]

    # Return feature score
    return HFC


def feature_spatial(fslDir, tempDir, aromaDir, melIC,
                    csf_mask="auto", out_mask="auto", edge_mask="auto"):
    """
    This function extracts the spatial feature scores. For each IC it
    determines the fraction of the mixture modeled thresholded Z-maps
    respectively located within the CSF or at the brain edges, using predefined
    standardized masks.

    Parameters
    ----------
    fslDir:     Full path of the bin-directory of FSL
    tempDir:    Full path of a directory where temporary files can be stored (called 'temp_IC.nii.gz')
    aromaDir:   Full path of the ICA-AROMA directory, containing the mask-files
    (mask_edge.nii.gz, mask_csf.nii.gz & mask_out.nii.gz)
    melIC:      Full path of the nii.gz file containing mixture-modeled
    thresholded (p>0.5) Z-maps, registered to the MNI152 2mm template

    Returns
    -------
    edgeFract:  Array of the edge fraction feature scores for the components of the melIC file
    csfFract:   Array of the CSF fraction feature scores for the components of the melIC file
    """

    # Get the number of ICs
    numICs = int(
        subprocess.getoutput(
            "{0} {1} | grep dim4 | head -n1 | awk '{print $2}'".format(
                op.join(fslDir, "fslinfo"), melIC
            )
        )
    )

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
    edgeFract = np.zeros(numICs)
    csfFract = np.zeros(numICs)
    for i in range(0, numICs):
        # Define temporary IC-file
        tempIC = op.join(tempDir, "temp_IC.nii.gz")

        # Extract IC from the merged melodic_IC_thr2MNI2mm file
        os.system(" ".join([op.join(fslDir, "fslroi"), melIC, tempIC, str(i), "1"]))

        # Change to absolute Z-values
        os.system(" ".join([op.join(fslDir, "fslmaths"), tempIC, "-abs", tempIC]))

        # Get sum of Z-values within the total Z-map (calculate via the mean
        # and number of non-zero voxels)
        totVox = int(
            subprocess.getoutput(
                "{0} {1} -V | awk '{print $1}'".format(
                    op.join(fslDir, "fslstats"), tempIC
                )
            )
        )

        if totVox != 0:
            totMean = float(
                subprocess.getoutput(
                    "{0} {1} -M".format(op.join(fslDir, "fslstats"), tempIC)
                )
            )
        else:
            print("     - The spatial map of component {0} is empty. Please check!".format(i + 1))
            totMean = 0

        totSum = totMean * totVox

        # Get sum of Z-values of the voxels located within the CSF
        # (calculate via the mean and number of non-zero voxels)
        csfVox = int(
            subprocess.getoutput(
                "{0} {1} -k {2} -V | awk '{print $1}'".format(
                    op.join(fslDir, "fslstats"), tempIC, csf_mask
                )
            )
        )
        if csfVox != 0:
            csfMean = float(
                subprocess.getoutput(
                    "{0} {1} -k {2} -M".format(
                        op.join(fslDir, "fslstats"), tempIC, csf_mask
                    )
                )
            )
        else:
            csfMean = 0

        csfSum = csfMean * csfVox

        # Get sum of Z-values of the voxels located within the Edge
        # (calculate via the mean and number of non-zero voxels)
        edgeVox = int(
            subprocess.getoutput(
                "{0} {1} -k {2} -V | awk '{print $1}'".format(
                    op.join(fslDir, "fslstats"), tempIC, edge_mask
                )
            )
        )
        if edgeVox != 0:
            edgeMean = float(
                subprocess.getoutput(
                    "{0} {1} -k {2} -M".format(
                        op.join(fslDir, "fslstats"), tempIC, edge_mask
                    )
                )
            )
        else:
            edgeMean = 0

        edgeSum = edgeMean * edgeVox

        # Get sum of Z-values of the voxels located outside the brain
        # (calculate via the mean and number of non-zero voxels)
        outVox = int(
            subprocess.getoutput(
                "{0} {1} -k {2} -V | awk '{print $1}'".format(
                    op.join(fslDir, "fslstats"), tempIC, out_mask
                )
            )
        )
        if outVox != 0:
            outMean = float(
                subprocess.getoutput(
                    "{0} {1} -k {2} -M".format(
                        op.join(fslDir, "fslstats"), tempIC, out_mask
                    )
                )
            )
        else:
            outMean = 0

        outSum = outMean * outVox

        # Determine edge and CSF fraction
        if totSum != 0:
            edgeFract[i] = (outSum + edgeSum) / (totSum - csfSum)
            csfFract[i] = csfSum / totSum
        else:
            edgeFract[i] = 0
            csfFract[i] = 0

    # Remove the temporary IC-file
    os.remove(tempIC)

    # Return feature scores
    return edgeFract, csfFract
