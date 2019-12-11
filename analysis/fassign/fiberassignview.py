#!/usr/bin/env python
"""Scriptable module to plot fiberfluxes in a fiberassign tile. Can
also be imported as a module in a notebook. See command line use below.

usage: fiberassignview.py [-h] [-b BAND] [-t TITLE] [-o OUTPUT] [-d] fafile

Fiberassign tile plotter.

positional arguments:
  fafile                Fiberassign file (full path)

optional arguments:
  -h, --help            show this help message and exit
  -b BAND, --band BAND  Filter to use [g, r, z].
  -t TITLE, --title TITLE
                        Plot title.
  -o OUTPUT, --output OUTPUT
                        Output image file name.
  -d, --display         If enabled, plot output to screen.
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

plt.style.use({'font.size' : 14})

try:
    from desitarget.sv1 import sv1_targetmask
    list_targets = True
except ImportError as e:
    list_targets = False
    print(e)


def plot_fiberflux(fafile, band='z', title=''):
    """Plot fiberfluxes for targets in a fiberassign file.

    Parameters
    ----------
    fafile : str
        Full path to a fiberassign FITS file.
    band : str
        Filter for plotting (g, r, or z).
    title : str
        Title to use in plot.

    Returns
    -------
    fig : mpl.Figure
        Matplotlib figure object.
    """
    hdus = fits.open(fafile)
    fa = hdus['FIBERASSIGN'].data

    # Read xy positions and fiberflux in a given band.
    x, y = fa['FIBERASSIGN_X'], fa['FIBERASSIGN_Y']

    # Check for valid band, recover if bad input.
    if band.upper() not in 'GRZ':
        print("Can't use band {}; defaulting to Z.".format(band))

    # Convert to magnitude (fiberflux is in nanomaggies).
    ffl = fa['FIBERFLUX_{}'.format(band.upper())]
    logf = np.full(len(ffl), -np.inf)
    logf[ffl > 0] = np.log10(ffl[ffl > 0])
    s = np.zeros_like(logf)
    m = 22.5 - 2.5*logf

    # Some junk to set scatterplot marker size.
    s = np.zeros_like(m)
    s[np.isfinite(m)] = (36 - m[np.isfinite(m)])

    fig, ax = plt.subplots(1,1, figsize=(9,7), tight_layout=True)
    ax.plot(x, y, ',')
    sc = ax.scatter(x, y, s=s, c=m, marker='o', cmap='magma', alpha=0.5, vmin=10, vmax=36)
    cb = fig.colorbar(sc)
    cb.set_label(r'$m_\mathrm{fiber}$')
    ax.set(aspect='equal',
           xlim=(-450,450),
           xlabel='$x$ [mm]',
           ylim=(-450,450),
           ylabel='$y$ [mm]',
           title=title)
    plt.xticks(rotation=45)

    return fig


def list_desi_targets(fafile):
    """List DESI_TARGET names in a fiberassign file.

    Parameters
    ----------
    fafile : str
        Full path to a fiberassign FITS file.
    """
    # DATA
    #     bgs_mask = sv1_bgs_mask:
    #       - [BGS_FAINT,        0, "BGS fai...': 2, 'O...
    #     desi_mask = sv1_desi_mask:
    #       - [LRG,              0, "LRG", ...T12|TWI...
    #     mws_mask = sv1_mws_mask:
    #       - [MWS_MAIN,         0, "Milky W... 'MORE_Z...
    #     obsmask = sv1_obsmask:
    #       - [UNOBS,            0, "unobserv... not obse...
    #     scnd_mask = sv1_scnd_mask:
    #       - [VETO,             0, "Never ...MORE_ZW...
    hdus = fits.open(fafile)
    desi_targets = hdus['TARGETS'].data['DESI_TARGET']

    names = []
    for dtgt in desi_targets:
        nm = sv1_targetmask.desi_mask.names(dtgt)
        if nm:
            names.append(nm)
      
    nm, count = np.unique(names, return_counts=True)
    for n_, c_ in zip(nm, count):
        print('{:20s} - {:d} targets'.format(n_, c_))


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Fiberassign tile plotter.')
    parser.add_argument('fafile', nargs=1, help='Fiberassign file (full path)')
    parser.add_argument('-b', '--band', dest='band', default='z',
                        help='Filter to use [g, r, z].')
    parser.add_argument('-t', '--title', dest='title', default='',
                        help='Plot title.')
    parser.add_argument('-o', '--output', dest='output', default=None,
                        help='Output image file name.')
    parser.add_argument('-d', '--display', dest='display',
                        action='store_true', default=False,
                        help='If enabled, plot output to screen.')
    args = parser.parse_args()

    fafile = args.fafile[0]

    if list_targets:
        list_desi_targets(fafile)

    fig = plot_fiberflux(fafile, band=args.band, title=args.title)

    if args.output is not None:
        fig.savefig(args.output)

    if args.display:
        plt.show()
