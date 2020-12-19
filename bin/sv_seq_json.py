#!/usr/bin/env python
"""Generate ICS DESI sequences for observing Survey Validation tiles, which can
require long runs of exposurse on the same location. This script lets the
observer quickly adjust the length and nubmer of exposures and specify the
observing program information.

See additional details at

https://desi.lbl.gov/trac/wiki/DESIOperations/SVSeq
"""

import os
import json
import logging as log
import numpy as np
from astropy import units as u
from astropy.io import fits
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_tile_coords(tileid, dryrun=False):
    """Open fiberassign file and read tile location.

    Parameters
    ----------
    tileid : int
        Tile ID.
    dryrun : bool
        If true, return dummy values.

    Returns
    -------
    ra : float
        Boresight RA for this tile.
    dec : float
        Boresight declination for this tile.
    """

    if dryrun:
        return 0., 0.

    if 'DOS_DESI_TILES' in os.environ:
        path = os.environ['DOS_DESI_TILES']
    else:
        # Fallback to hardcoded path (issue a warning).
        path = '/data/tiles/ALL_tiles/20191119'
        log.warning('DOS_DESI_TILES undefined; using {}'.format(path))

    # Warning (SB): subfolder in DOS_DESI_TILES likely to change.
    fibfile = '/'.join([path, '080', 'fiberassign-{:06d}.fits'.format(tileid)])
    try:
        hdus = fits.open(fibfile)
    except FileNotFoundError as e:
        log.error('TILE {} cannot be found.\n{}.'.format(tileid, e))
        raise SystemExit

    log.debug('Reading {}'.format(fibfile))
    header = hdus['PRIMARY'].header
    ra, dec = header['TILERA'], header['TILEDEC']
    return ra, dec


def is_bright(tile_type):
    """Set up and write out JSON script for an SV observing sequence.

    Parameters
    ----------
    tile_type : str
        A tile type string such as ELG, QSO+LRG, BGS+MWS, etc.
    """
    return 'BGS' in tile_type.upper() or 'MWS' in tile_type.upper() or 'BRIGHT' in tile_type.upper()


def setup_sequence(args):
    """Set up and write out JSON script for an SV observing sequence.

    Parameters
    ----------
    args : argparse.Namespace
        Argument list from command line fibermode subprogram.
    """
    # Generate single interleaved DESI sequence + spectrograph script:
    sv_script = []

    tile_id = args.tile_id[0]
    sv_version = args.svversion
    tile_type = args.tiletype

    if is_bright(tile_type):
        if args.exptime > 300. and not args.override_bright:
            raise ValueError('CAUTION: attempted >300 s exposure for a BRIGHT tile!\n'
                             'If desired, enable with --override-bright')


    log.debug('{:>7s} {:>7s} {:>7s}'.format('Tile', 'RA', 'Dec'))

    tile_ra, tile_dec = get_tile_coords(tile_id, args.dryrun)
    log.debug('{:7d} {:7g} {:7g}'.format(tile_id, tile_ra, tile_dec))

    # Logging variables sent to ICS for output to FITS headers:
    passthru = '{{ TILEID:{:d}, TILERA:{:g}, TILEDEC:{:g} }}'.format(tile_id, tile_ra, tile_dec)

    # Stack up DESI sequences. Note: exptime is for spectrographs.
    for i in range(args.nexposures):
        sv_script.append({'sequence'            : 'DESI',
                           'fiberassign'         : tile_id,
                           'exptime'             : args.exptime,
                           'guider_exptime'      : 5.0,
                           'acquisition_exptime' : 15.0,
                           'fvc_exptime'         : 2.0,
                           'usefocus'            : True,
                           'usesky'              : True,
                           'usedonut'            : True,
                           'focus_exptime'       : 60.0,
                           'sky_exptime'         : 60.0,
                           'program'             : 'SV{:d} {} tile {:d}'.format(sv_version, tile_type, tile_id)})

    # Dump JSON DESI sequence list into a file.
    filename = 'seq_SV{:d}_tile{:06d}_{}_{:d}x{:.0f}s.json'.format(sv_version,
        tile_id,
        tile_type.replace('+','_').replace('*','_').replace(' ','_'),
        args.nexposures,
        args.exptime,
        )

    json.dump(sv_script, open(filename, 'w'), indent=4)
    log.info('Use {} in the DESI observer console.'.format(filename))


if __name__ == '__main__':
    # Main options specified before rastermode and fibermode subcommands.
    p = ArgumentParser(description='Survey Validation sequence JSON ICS setup program',
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-d', '--dryrun', action='store_true', default=False,
                   help='Dry run: do not check for fiberassign files')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Verbose output for debugging')
    p.add_argument('-e', '--exptime', type=float, default=900.0,
                   help='Exposure time [seconds]')
    p.add_argument('-n', '--nexposures', type=int, default=4,
                   help='Number of exposures per tile.')
    p.add_argument('-s', '--svversion', type=int, default=1,
                   help='Survey Validation version ID.')
    p.add_argument('-t', '--tiletype', required=True,
                   help='QSO+LRG, ELG, BGS+MWS.')
    p.add_argument('--override-bright', dest='override_bright', action='store_true',
                   help='Override bright time exposure restriction.')
    p.add_argument('tile_id', nargs=1, type=int,
                   help='Tile ID for a sequence of SV[n] exposures.')
    p.set_defaults(func=setup_sequence)

    args = p.parse_args()

    if 'func' in args:
        if args.verbose:
            log.basicConfig(format='%(levelname)s: %(message)s', level=log.DEBUG)
        else:
            log.basicConfig(format='%(levelname)s: %(message)s', level=log.INFO)
        args.func(args)
    else:
        p.print_help()
