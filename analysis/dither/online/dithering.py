#!/usr/bin/env python
"""Module to read and analyze spectrograph exposures for dither sequences.
Accesses the data from Nightwatch.
"""

import os
import numpy as np
from glob import glob
from astropy.table import Table, join
from find_exposures import find_exposures


def get_qafile(date, exp_id):
    """Get the Nightwatch QA file for a given date and exposure ID.
    
    Parameters
    ----------
    date : str
        Observation date [YYYYMMDD]
    exp_id : int
        Exposure ID

    Returns
    -------
    filename : str
        Path to Nightwatch qa-*.fits file.
    """
    filename = '/exposures/nightwatch/{}/{:08d}/qa-{:08d}.fits'.format(date, exp_id, exp_id)
    if os.path.isfile(filename):
        return filename
    return None


def get_pmx_fiberassign_table(date, exp_id):
    """Get the fiberassign table for a given date and PMX exposure ID.
    
    Parameters
    ----------
    date : str
        Observation date [YYYYMMDD]
    exp_id : int
        Exposure ID

    Returns
    -------
    tab : astropy.Table
        Table with fiberassign values.
    """
    filename = glob('/exposures/desi/{}/{:08d}/fiberassign*.fits'.format(
        date, exp_id))

    tab = None
    if filename:
        tab = Table.read(filename[0], hdu='FASSIGN')
        tab = tab['FIBER', 'TARGETID', 'LOCATION', 'FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'FIBERASSIGN_X', 'FIBERASSIGN_Y']
    return tab


def get_camfiber_table(date, exp_id):
    """Get the Nightwatch qafile camfiber table for a given date and spectro
    exposure ID.
    
    Parameters
    ----------
    date : str
        Observation date [YYYYMMDD]
    exp_id : int
        Exposure ID

    Returns
    -------
    tab : astropy.Table
        Table with camfiber data.
    """
    filename = '/exposures/nightwatch/{}/{:08d}/qa-{:08d}.fits'.format(date, exp_id, exp_id)

    tab = None
    if os.path.isfile(filename):
        tab = Table.read(filename, hdu='PER_CAMFIBER')
        tab = tab['FIBER', 'MEDIAN_CALIB_SNR', 'CAM']
    return tab


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from argparse import ArgumentParser

    parser = ArgumentParser(description='DOS exposure finder')
    parser.add_argument('-p', '--pmxid', dest='pmx_id', nargs=1, default=None,
                        help='PMX acquisition sequence')
    parser.add_argument('exprange', nargs=2, type=int,
                        help='Exposure ID range for dither sequence')
    args = parser.parse_args()

    # Get PMX fiberassign data.
    if args.pmx_id is None:
        parser.print_help()
        raise SystemExit('\nMust provide a PMX exposure ID')

    records = find_exposures(exp_id=args.pmx_id, seq='PMX')
    id_, date_, seq_, prog_ = records[0]
    fassign_tab = get_pmx_fiberassign_table(date_, id_)

    # Get Nightwatch qa-{000nnnnn}.fits exposure data.
    exmin, exmax = args.exprange
    spec_ids = np.arange(exmin, exmax+1)
    spec_recs = find_exposures(exp_id=spec_ids, seq='Spectrographs')

    # Extract all data and prepare some plots.
    camfiber_tabs = {}
    max_snr = -1e99

    for rec in spec_recs:
        # Get PER_CAMFIBER table from Nightwatch
        id_, date_, seq_, prog_ = rec
        camfiber_tab = get_camfiber_table(date_, id_)
        
        # Use only the B camera.
        bmask = camfiber_tab['CAM'] == 'B'
        
        # Join PER_CAMFIBER table with FASSIGN table.
        camfiber_tab = join(camfiber_tab[bmask], fassign_tab, keys='FIBER')
        camfiber_tabs[id_] = camfiber_tab
        
        # Store the max_snr across all exposures, which is handy for plotting down below.
        max_snr = np.maximum(max_snr, np.max(camfiber_tab['MEDIAN_CALIB_SNR']))

    fig, axes = plt.subplots(4,3, figsize=(6.5*3, 5*4), sharex=True, sharey=True)
    axes = axes.flatten()

    n = len(axes)
    m = len(camfiber_tabs)
    if n > m:
        for i in range(m, n):
            axes[i].set_visible(False)  

    for i, (exp_id, data) in enumerate(camfiber_tabs.items()):
        ax = axes[i]
        
        x = data['FIBERASSIGN_X']
        y = data['FIBERASSIGN_Y']
        snr = data['MEDIAN_CALIB_SNR']
        snr_cut = snr > 0.5

        ax.plot(x, y, ',')
        sc = ax.scatter(x[snr_cut], y[snr_cut], c=snr[snr_cut], cmap='plasma', vmax=max_snr)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label('median calib SNR')

        ax.set(aspect='equal',
               xlabel='$x$',
               ylabel='$y$',
               title='Exposure {}'.format(exp_id))  
        
    fig.tight_layout()
    fig.savefig('raster_snr.png', dpi=100)

