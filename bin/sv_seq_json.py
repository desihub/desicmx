#!/usr/bin/env python
"""Generate guider and spectrograph ICS sequences for running telescope dither
tests. Can run in two modes:

1. rastermode, which will move the telescope boresight given a tile ID and step
size. One can specify an offset of the raster center from the current location;
the script will return to the current location at the end. And the labels on
the raster positions will be relative to the current location. This is so that
one can do multiple rasters in the same coordinate system.

2. fibermode, which will move the positioners given a range of consecutive tile
IDs.  

Based on instructions available at

https://desi.lbl.gov/trac/wiki/DESIOperations/DithSeq
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

    header = hdus['PRIMARY'].header
    ra, dec = header['TILERA'], header['TILEDEC']
    return ra, dec


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
                           'program'             : 'SV{:d} {} tile {:d} ({:g}, {:g})'.format(sv_version, tile_type, tile_id, tile_ra, tile_dec)})

    # Dump JSON DESI sequence list into a file.
    filename = 'seq_SV{:d}_{}_tile{:06d}_{:d}x{:.0f}s.json'.format(sv_version,
        tile_type.replace('+','_').replace('*','_'),
        tile_id,
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
                   help='QSO+LRG, ELG, BGS.')
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
