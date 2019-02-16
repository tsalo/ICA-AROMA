from __future__ import print_function

import argparse

from aroma import plotting


def _get_parser():
    parser = argparse.ArgumentParser(
        description=("Plot component classification overview similar to plot "
                     "in the main AROMA paper")
    )
    # Required options
    reqoptions = parser.add_argument_group("Required arguments")
    reqoptions.add_argument(
        "-i",
        "-in",
        dest="myinput",
        required=True,
        help="Input query or filename. Use quotes when specifying a query",
    )

    optoptions = parser.add_argument_group("Optional arguments")
    optoptions.add_argument(
        "-outdir",
        dest="outDir",
        required=False,
        default=".",
        help="Specification of directory where figure will be saved",
    )
    optoptions.add_argument(
        "-type",
        dest="plottype",
        required=False,
        default="assessment",
        options=["assessment"],
        help=("Specification of the type of plot you want. Currently this is "
              "a placeholder option for potential other plots that might be "
              "added in the future."),
    )
    return parser


def _main(argv=None):
    """Tedana entry point"""
    options = _get_parser().parse_args(argv)
    main(**vars(options))


def main(myinput, outDir, plottype="assessment"):
    if plottype == "assessment":
        plotting.classification_plot(myinput, outDir)


if __name__ == "__main__":
    _main()
