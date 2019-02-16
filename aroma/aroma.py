#!/usr/bin/env python
"""
Functions for ICA-AROMA v0.3 beta
"""
from __future__ import division
from __future__ import print_function

import os
import os.path as op
import subprocess

import numpy as np


def runICA(fslDir, inFile, outDir, melDirIn, mask, dim, TR):
    """
    This function runs MELODIC and merges the mixture modeled thresholded
    ICs into a single 4D nifti file

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    inFile:     Full path to the fMRI data file (nii.gz) on which MELODIC should be run
    outDir:     Full path of the output directory
    melDirIn:   Full path of the MELODIC directory in case it has been run
    before, otherwise define empty string
    mask:       Full path of the mask to be applied during MELODIC
    dim:        Dimensionality of ICA
    TR:     TR (in seconds) of the fMRI data

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    melodic.ica     MELODIC directory
    melodic_IC_thr.nii.gz   merged file containing the mixture modeling
    thresholded Z-statistical maps located in melodic.ica/stats/
    """

    # Define the 'new' MELODIC directory and predefine some associated files
    melDir = op.join(outDir, "melodic.ica")
    melIC = op.join(melDir, "melodic_IC.nii.gz")
    melICmix = op.join(melDir, "melodic_mix")
    melICthr = op.join(outDir, "melodic_IC_thr.nii.gz")

    # When a MELODIC directory is specified,
    # check whether all needed files are present.
    # Otherwise... run MELODIC again
    if (
        len(melDir) != 0
        and op.isfile(op.join(melDirIn, "melodic_IC.nii.gz"))
        and op.isfile(op.join(melDirIn, "melodic_FTmix"))
        and op.isfile(op.join(melDirIn, "melodic_mix"))
    ):

        print("  - The existing/specified MELODIC directory will be used.")

        # If a 'stats' directory is present (contains thresholded spatial maps)
        # create a symbolic link to the MELODIC directory.
        # Otherwise create specific links and
        # run mixture modeling to obtain thresholded maps.
        if op.isdir(op.join(melDirIn, "stats")):
            os.symlink(melDirIn, melDir)
        else:
            print(
                "  - The MELODIC directory does not contain the required "
                "'stats' folder. Mixture modeling on the Z-statistical maps "
                "will be run."
            )

            # Create symbolic links to the items in the specified melodic directory
            os.makedirs(melDir)
            for item in os.listdir(melDirIn):
                os.symlink(op.join(melDirIn, item), op.join(melDir, item))

            # Run mixture modeling
            os.system(
                " ".join(
                    [
                        op.join(fslDir, "melodic"),
                        "--in=" + melIC,
                        "--ICs=" + melIC,
                        "--mix=" + melICmix,
                        "--outdir=" + melDir,
                        "--Ostats --mmthresh=0.5",
                    ]
                )
            )

    else:
        # If a melodic directory was specified, display that it did not contain
        # all files needed for ICA-AROMA (or that the directory does not exist
        # at all)
        if len(melDirIn) != 0:
            if not op.isdir(melDirIn):
                print(
                    "  - The specified MELODIC directory does not exist. MELODIC will be run seperately."
                )
            else:
                print(
                    "  - The specified MELODIC directory does not contain "
                    "the required files to run ICA-AROMA. MELODIC will be "
                    "run seperately."
                )

        # Run MELODIC
        os.system(
            " ".join(
                [
                    op.join(fslDir, "melodic"),
                    "--in=" + inFile,
                    "--outdir=" + melDir,
                    "--mask=" + mask,
                    "--dim=" + str(dim),
                    "--Ostats --nobet --mmthresh=0.5 --report",
                    "--tr=" + str(TR),
                ]
            )
        )

    # Get number of components
    cmd = " ".join(
        [op.join(fslDir, "fslinfo"), melIC, "| grep dim4 | head -n1 | awk '{print $2}'"]
    )
    nrICs = int(float(subprocess.getoutput(cmd)))

    # Merge mixture modeled thresholded spatial maps. Note! In case that
    # mixture modeling did not converge, the file will contain two spatial
    # maps. The latter being the results from a simple null hypothesis test.
    # In that case, this map will have to be used (first one will be empty).
    for i in range(1, nrICs + 1):
        # Define thresholded zstat-map file
        zTemp = op.join(melDir, "stats", "thresh_zstat" + str(i) + ".nii.gz")
        cmd = " ".join(
            [
                op.join(fslDir, "fslinfo"),
                zTemp,
                "| grep dim4 | head -n1 | awk '{print $2}'",
            ]
        )
        lenIC = int(float(subprocess.getoutput(cmd)))

        # Define zeropad for this IC-number and new zstat file
        cmd = " ".join([op.join(fslDir, "zeropad"), str(i), "4"])
        ICnum = subprocess.getoutput(cmd)
        zstat = op.join(outDir, "thr_zstat" + ICnum)

        # Extract last spatial map within the thresh_zstat file
        os.system(
            " ".join(
                [
                    op.join(fslDir, "fslroi"),
                    zTemp,  # input
                    zstat,  # output
                    str(lenIC - 1),  # first frame
                    "1",
                ]
            )
        )  # number of frames

    # Merge and subsequently remove all mixture modeled Z-maps within the output directory
    os.system(
        " ".join(
            [
                op.join(fslDir, "fslmerge"),
                "-t",  # concatenate in time
                melICthr,  # output
                op.join(outDir, "thr_zstat????.nii.gz"),
            ]
        )
    )  # inputs

    os.system("rm " + op.join(outDir, "thr_zstat????.nii.gz"))

    # Apply the mask to the merged file (in case a melodic-directory was
    # predefined and run with a different mask)
    os.system(
        " ".join([op.join(fslDir, "fslmaths"), melICthr, "-mas " + mask, melICthr])
    )


def register2MNI(fslDir, inFile, outFile, affmat, warp):
    """
    This function registers an image (or time-series of images) to MNI152 T1
    2mm. If no affmat is defined, it only warps (i.e. it assumes that the data
    has been registerd to the structural scan associated with the warp-file
    already). If no warp is defined either, it only resamples the data to 2mm
    isotropic if needed (i.e. it assumes that the data has been registered to a
    MNI152 template). In case only an affmat file is defined, it assumes that
    the data has to be linearly registered to MNI152 (i.e. the user has a
    reason not to use non-linear registration on the data).

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    inFile:     Full path to the data file (nii.gz) which has to be registerd to MNI152 T1 2mm
    outFile:    Full path of the output file
    affmat:     Full path of the mat file describing the linear registration
    (if data is still in native space)
    warp:       Full path of the warp file describing the non-linear
    registration (if data has not been registered to MNI152 space yet)

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    melodic_IC_mm_MNI2mm.nii.gz merged file containing the mixture modeling
    thresholded Z-statistical maps registered to MNI152 2mm
    """

    # Define the MNI152 T1 2mm template
    fslnobin = fslDir.rsplit("/", 2)[0]
    ref = op.join(fslnobin, "data", "standard", "MNI152_T1_2mm_brain.nii.gz")

    # If the no affmat- or warp-file has been specified, assume that the data
    # is already in MNI152 space. In that case only check if resampling to 2mm
    # is needed
    if (len(affmat) == 0) and (len(warp) == 0):
        # Get 3D voxel size
        pixdim1 = float(
            subprocess.getoutput(
                "%sfslinfo %s | grep pixdim1 | awk '{print $2}'" % (fslDir, inFile)
            )
        )
        pixdim2 = float(
            subprocess.getoutput(
                "%sfslinfo %s | grep pixdim2 | awk '{print $2}'" % (fslDir, inFile)
            )
        )
        pixdim3 = float(
            subprocess.getoutput(
                "%sfslinfo %s | grep pixdim3 | awk '{print $2}'" % (fslDir, inFile)
            )
        )

        # If voxel size is not 2mm isotropic, resample the data, otherwise copy the file
        if (pixdim1 != 2) or (pixdim2 != 2) or (pixdim3 != 2):
            os.system(
                " ".join(
                    [
                        op.join(fslDir, "flirt"),
                        " -ref " + ref,
                        " -in " + inFile,
                        " -out " + outFile,
                        " -applyisoxfm 2 -interp trilinear",
                    ]
                )
            )
        else:
            os.system("cp " + inFile + " " + outFile)

    # If only a warp-file has been specified, assume that the data has already
    # been registered to the structural scan. In that case apply the warping
    # without a affmat
    elif (len(affmat) == 0) and (len(warp) != 0):
        # Apply warp
        os.system(
            " ".join(
                [
                    op.join(fslDir, "applywarp"),
                    "--ref=" + ref,
                    "--in=" + inFile,
                    "--out=" + outFile,
                    "--warp=" + warp,
                    "--interp=trilinear",
                ]
            )
        )

    # If only a affmat-file has been specified perform affine registration to MNI
    elif (len(affmat) != 0) and (len(warp) == 0):
        os.system(
            " ".join(
                [
                    op.join(fslDir, "flirt"),
                    "-ref " + ref,
                    "-in " + inFile,
                    "-out " + outFile,
                    "-applyxfm -init " + affmat,
                    "-interp trilinear",
                ]
            )
        )

    # If both a affmat- and warp-file have been defined, apply the warping accordingly
    else:
        os.system(
            " ".join(
                [
                    op.join(fslDir, "applywarp"),
                    "--ref=" + ref,
                    "--in=" + inFile,
                    "--out=" + outFile,
                    "--warp=" + warp,
                    "--premat=" + affmat,
                    "--interp=trilinear",
                ]
            )
        )


def classification(maxRPcorr, edgeFract, HFC, csfFract):
    """
    This function classifies a set of components into motion and
    non-motion components based on four features;
    maximum RP correlation, high-frequency content, edge-fraction and CSF-fraction

    Parameters
    ---------------------------------------------------------------------------------
    outDir:     Full path of the output directory
    maxRPcorr:  Array of the 'maximum RP correlation' feature scores of the components
    edgeFract:  Array of the 'edge fraction' feature scores of the components
    HFC:        Array of the 'high-frequency content' feature scores of the components
    csfFract:   Array of the 'CSF fraction' feature scores of the components

    Return
    ---------------------------------------------------------------------------------
    motionICs   Array containing the indices of the components identified as motion components

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    classified_motion_ICs.txt   A text file containing the indices of the
    components identified as motion components
    """
    assert len(maxRPcorr) == len(edgeFract) == len(csfFract) == len(HPC)
    # Classify the ICs as motion or non-motion

    # Define criteria needed for classification (thresholds and hyperplane-parameters)
    thr_csf = 0.10
    thr_HFC = 0.35
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

    # Project edge & maxRPcorr feature scores to new 1D space
    x = np.array([maxRPcorr, edgeFract])
    proj = hyp[0] + np.dot(x.T, hyp[1:])

    # Classify the ICs
    motionICs = np.squeeze(
        np.array(
            np.where((proj > 0) + (csfFract > thr_csf) + (HFC > thr_HFC))
        )
    )

    return motionICs


def denoising(fslDir, inFile, outDir, melmix, denType, denIdx):
    """
    This function classifies the ICs based on the four features;
    maximum RP correlation, high-frequency content, edge-fraction and CSF-fraction

    Parameters
    ---------------------------------------------------------------------------------
    fslDir:     Full path of the bin-directory of FSL
    inFile:     Full path to the data file (nii.gz) which has to be denoised
    outDir:     Full path of the output directory
    melmix:     Full path of the melodic_mix text file
    denType:    Type of requested denoising ('aggr': aggressive, 'nonaggr':
    non-aggressive, 'both': both aggressive and non-aggressive
    denIdx:     Indices of the components that should be regressed out

    Output (within the requested output directory)
    ---------------------------------------------------------------------------------
    denoised_func_data_<denType>.nii.gz:        A nii.gz file of the denoised fMRI data
    """

    # Import required modules
    import os
    import numpy as np

    # Check if denoising is needed (i.e. are there components classified as motion)
    check = denIdx.size > 0

    if check == 1:
        # Put IC indices into a char array
        if denIdx.size == 1:
            denIdxStrJoin = "%d" % (denIdx + 1)
        else:
            denIdxStr = np.char.mod("%i", (denIdx + 1))
            denIdxStrJoin = ",".join(denIdxStr)

        # Non-aggressive denoising of the data using fsl_regfilt (partial regression), if requested
        if (denType == "nonaggr") or (denType == "both"):
            os.system(
                " ".join(
                    [
                        op.join(fslDir, "fsl_regfilt"),
                        "--in=" + inFile,
                        "--design=" + melmix,
                        '--filter="' + denIdxStrJoin + '"',
                        "--out=" + op.join(outDir, "denoised_func_data_nonaggr.nii.gz"),
                    ]
                )
            )

        # Aggressive denoising of the data using fsl_regfilt (full regression)
        if (denType == "aggr") or (denType == "both"):
            os.system(
                " ".join(
                    [
                        op.join(fslDir, "fsl_regfilt"),
                        "--in=" + inFile,
                        "--design=" + melmix,
                        '--filter="' + denIdxStrJoin + '"',
                        "--out=" + op.join(outDir, "denoised_func_data_aggr.nii.gz"),
                        "-a",
                    ]
                )
            )
    else:
        print(
            "  - None of the components were classified as motion, so no "
            "denoising is applied (a symbolic link to the input file will "
            "be created)."
        )
        if (denType == "nonaggr") or (denType == "both"):
            os.symlink(inFile, op.join(outDir, "denoised_func_data_nonaggr.nii.gz"))
        if (denType == "aggr") or (denType == "both"):
            os.symlink(inFile, op.join(outDir, "denoised_func_data_aggr.nii.gz"))
