"""
Tools for setting up nice Jupyter notebooks.
"""
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython


def magic(text):
    ipy = get_ipython()
    if ipy is not None:
        ipy.magic(text)


def configure_mpl_fast():
    """
    No retina and no latex, fast matplotlib. This can be useful for making
    lots of complex plots. The "pretty" plots often take 5x longer.
    """
    magic("config InlineBackend.figure_format='png'")


def configure_mpl_pretty():
    """Retina and Latex matplotlib figures"""
    magic("config InlineBackend.figure_format='retina'")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def setup_nb(text_size_ratio=0.75, pretty=True, autoreload=True):
    """
    This function is handy to call at the top of a Jupyter notebook. It sets up:
    1. autoreload for allowing python modules to be modified without restart the
       notebook.
    2. sane matplotlib defaults for making good looking figures including:
        - retina mode
        - good colors
        - solid non-transparent background
        - nice text sizes
    """
    if autoreload:
        magic("load_ext autoreload")
        magic("autoreload 2")

    if pretty:
        configure_mpl_pretty()
    else:
        configure_mpl_fast()

    plt.rcParams["axes.facecolor"] = (1.0, 1.0, 1.0, 1.0)
    plt.rcParams["figure.facecolor"] = (1.0, 1.0, 1.0, 1.0)
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["image.cmap"] = "plasma"
    scale_text(factor=text_size_ratio)
    np.set_printoptions(edgeitems=10, linewidth=100)


def scale_text(factor=0.75):
    plt.rcParams["font.size"] = 20 * factor
    plt.rcParams["axes.labelsize"] = 18 * factor
    plt.rcParams["axes.titlesize"] = 20 * factor
    plt.rcParams["xtick.labelsize"] = 16 * factor
    plt.rcParams["ytick.labelsize"] = 16 * factor
    plt.rcParams["legend.fontsize"] = 20 * factor
    plt.rcParams["figure.titlesize"] = 22 * factor


def contourf_tbt(x, y, z, levels=None):
    cntf = plt.contourf(x, y, z, levels=levels, extend="both")
    plt.contour(
        x,
        y,
        z,
        levels=levels,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        extend="both",
    )
    cbar = plt.colorbar(cntf)
    return cntf, cbar
