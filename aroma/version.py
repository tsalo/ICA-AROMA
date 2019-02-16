from __future__ import absolute_import, division, print_function

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 4
_version_micro = 4  # use '' for first of series, number for 1 and above
_version_extra = 'beta'

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = ("ICA-AROMA: a data-driven method to identify and remove head "
               "motion-related artefacts from functional MRI data.")
# Long description will go up on the pypi page
long_description = """

ICA-AROMA
=========

ICA-AROMA (i.e. ‘ICA-based Automatic Removal Of Motion Artifacts’) concerns a
data-driven method to identify and remove motion-related independent components
from fMRI data. To that end it exploits a small, but robust set of
theoretically motivated features, preventing the need for classifier
re-training and therefore providing direct and easy applicability.

License
=======
``NiMARE`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2015--, Maarten Mennes

"""

NAME = "ICA-AROMA"
MAINTAINER = "Maarten Mennes"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/maartenmennes/ICA-AROMA"
DOWNLOAD_URL = "https://github.com/maartenmennes/ICA-AROMA.git"
LICENSE = "Apache 2.0"
AUTHOR = "Maarten Mennes"
AUTHOR_EMAIL = "https://github.com/maartenmennes/ICA-AROMA"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
REQUIRES = ["nibabel", "numpy", "scipy", "pandas", "nipype",
            "scikit-learn", "matplotlib", "seaborn", "argparse"],
ENTRY_POINTS = {'console_scripts': [
    'aroma=aroma.cli.fsl_aroma:_main',
    'plot_aroma=aroma.cli.make_figures:_main']}
