#!/usr/bin/env python
"""Given a range of exposures for a dithering sequence, match the exposure IDs
from the spectrographs with the corresponding GFA exposures. Needs to be run on
a system with Nightwatch QA files and GFA files, and should work automatically
at KPNO or NERSC.
"""

import os
import socket
import warnings
import numpy as np
from glob import glob
from datetime import datetime
from argparse import ArgumentParser


def get_prefix():
    """Set the filesystem prefix at KPNO or NERSC.

    Returns
    -------
    prefix_gfas : str
        Path to GFA files on this filesystem.
    prefix_spec : str
        Path to Nightwatch files on this system.
    """
    if 'nersc' in socket.getfqdn():
        # Offline paths.
        prefix_gfas = '/global/project/projectdirs/desi/spectro/data'
        prefix_spec = '/global/project/projectdirs/desi/spectro/nightwatch/kpno'
    else:
        # Online paths.
        prefix_gfas = '/exposures/desi'
        prefix_spec = '/exposures/nightwatch'

    return prefix_gfas, prefix_spec


def get_exposure_sequence(date, expi, expj=None):
    """Access GFA and spectrograph exposures for a dither sequence. Try to
    run only on a single 3x3 raster scan at once, not a larger collection
    of dither sequences.
    
    Parameters
    ----------
    date : int or str
        Date of exposure(s), in format YYYYMMDD.
    expi : int or list of ints
        First exp ID in a range of GFA+spectrograph dithering exposures,
        or a list of spectrograph exposure IDs.
    expj : int
        Last exp ID in a range of GFA+spectrograph dithering exposures.
        
    Returns
    -------
    seq_gfas : list of str
        List of files with GFA exposures in a dither sequence.
    seq_spec : list of str
        List of files with spectrograph exposures in a dither sequence.
    """
    # Decide which paths to use depending on the filesystem.
    prefix_gfas, prefix_spec = get_prefix()

    # Check for valid arguments and set up the exposure list.
    if type(expi) is int:
        if type(expj) is not None:
            if type(expj) is int:
                seq = np.arange(expi, expj+1)
            else:
                raise TypeError('Argument expj must be an integer.')
    elif type(expi) is np.ndarray or type(expi) is list:
        seq = expi
        if expj is not None:
            warnings.warn('Argument expi is an exposure list; ignoring argument expj.', UserWarning)
    else:
        raise TypeError('Argument expi must be an integer.')
        
    # Check that the paths to GFA and spectrograph exposures exist.
    paths = []
    for pf in [prefix_spec, prefix_gfas]:
        path = '/'.join([pf, date]) if type(date) is str else '{}/{:08d}'.format(pf, date)
        if not os.path.isdir(path):
            raise RuntimeError('Path {} not found.'.format(path))
        paths.append(path)
        
    path_spec, path_gfas = paths
    
    # Loop through the exposure list and identify spectrograph exposures
    # and corresponding GFA exposures. Assume the GFA exposure comes right
    # before the spectrograph exposure.
    seq_spec = []
    seq_gfas = []
    exp_spec = []
    exp_gfas = []

    for expid in seq:
        folder_spec = '{}/{:08d}'.format(path_spec, expid)
        # If we found a spec exposure, search for the matching GFA exposure.
        if os.path.isdir(folder_spec):
            
            # Start looking backwards from the current spectrograph exposure,
            # up to expid-10 for now.
            folder_gfas = ''
            for expid_gfas in range(expid-1, expid-10, -1):
                folder_gfas = '{}/{:08d}'.format(path_gfas, expid_gfas)
                if os.path.isdir(folder_gfas):
                    break
            
            # If we don't find a GFA file just before the spectrograph exposure,
            # omit the exposure from the sequence and produce a warning.
            if not os.path.isdir(folder_gfas):
                warnings.warn('Exp {} does not have a matching GFA exposure; ignoring.'.format(expid))
                continue

            # Search for Nightwatch qa frames and GFA FITS files.
            # Check that the files exist.
            qafiles  = glob('{}/qa-*.fits'.format(folder_spec))            
            gfafiles = glob('{}/gfa-*.fits.fz'.format(folder_gfas))
            
            if not qafiles:
                warnings.warn('No qaframe found for exp {}. Ignoring.'.format(expid))
                continue
            if not gfafiles:
                # Check if we're at the final spectrograph exposure. If so, match it
                # with the first in the sequence.
                if expid == seq[-1]:
                    expid_gfas = exp_gfas[0]
                    folder_gfas = '{}/{:08d}'.format(path_gfas, expid_gfas)
                    gfafiles = glob('{}/gfa-*.fits.fz'.format(folder_gfas))
                    if not gfafiles:
                        warnings.warn('No GFA exposure found for exp {}. Ignoring.'.format(expid))
                        continue
                else:
                    warnings.warn('No GFA exposure found for exp {}. Ignoring.'.format(expid))
                    continue
            
            seq_spec.append(qafiles[0])
            exp_spec.append(expid)
            
            seq_gfas.append(gfafiles[0])
            exp_gfas.append(expid_gfas)
    
    return exp_gfas, seq_gfas, exp_spec, seq_spec


if __name__ == '__main__':
    parser = ArgumentParser(description='GFA+spectrograph exposure match.')
    parser.add_argument('-d', '--date', dest='date',
                        default='20191126',
                        help='Starting observation date [YYYYMMDD]')
    parser.add_argument('-e', '--expid', nargs=2, dest='expid', type=int,
                        default=[30085, 30123],
                        help='Exposure ID range [EXPMIN, EXPMAX]')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='Verbose output')
    args = parser.parse_args()

    date = args.date
    exp1, exp2 = args.expid

    ex_gfas, file_gfas, ex_spec, file_spec = get_exposure_sequence(date, exp1, exp2)

    if args.verbose:
        print('{:^8s} {:^8s} {}'.format('GFA', 'QA', 'Files'))
        for i, j, gfa, spec in zip(ex_gfas, ex_spec, file_gfas, file_spec):
            print('{:^8d} {:^8d} {}\n{:>18s}{}'.format(i, j, gfa, '', spec))
    else:
        print('GFA  Exposures: {}'.format(' '.join([str(e) for e in ex_gfas])))
        print('Spec Exposures: {}'.format(' '.join([str(e) for e in ex_spec])))
