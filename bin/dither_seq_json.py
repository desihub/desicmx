#!/usr/bin/env python

import os
import json
import logging as log
import numpy as np
from astropy import units as u
from astropy.io import fits
from argparse import ArgumentParser

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

    fibfile = '/'.join([path, 'fiberassign-{:06d}.fits'.format(tileid)])
    try:
        hdus = fits.open(fibfile)
    except FileNotFoundError as e:
        log.error('TILE {} cannot be found.\n{}.'.format(tileid, e))
        raise SystemExit

    header = hdus['PRIMARY'].header
    ra, dec = header['TILERA'], header['TILEDEC']
    return ra, dec


def setup_rastermode(args):
    """Set up and write out JSON telescope raster script for dither tests.

    Parameters
    ----------
    args : argparse.Namespace
        Argument list from command line fibermode subprogram.
    """

    RA = args.deltara*u.arcsec
    DEC = args.deltadec*u.arcsec
    ra = RA
    dec = DEC

    # Set step size. Add a unit from astropy, does not change value.
    step = args.step*u.arcsec
    tile_id = args.tileid
    tile_ra, tile_dec = get_tile_coords(tile_id, args.dryrun)

    # Standard raster: 3x3 with 3 visits to (0,0), 11 exposures total.
    if args.pattern == '3x3':
        stepx = np.asarray([ 0,  1, -1, -1,  0,  1,  1,  0, -1, -1,  1])*step
        stepy = np.asarray([ 0,  1,  0,  0, -1,  0,  0, -1,  0,  0,  1])*step

    # Extended raster: 5x5 with 3 visits to (0,0), 27 exposures in total.
    elif args.pattern == '5x5':
        # Inner 3x3:
        stepx = np.asarray([ 0,  1, -1, -1,  0,  0,  1,  1,  0, -1])
        stepy = np.asarray([ 0,  1,  0,  0, -1, -1,  0,  0,  1,  0])

        # Outer 5x5:
        _stepx = np.asarray([ 2, -1, -1, -1, -1,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0, -2])
        _stepy = np.asarray([ 2,  0,  0,  0,  0, -1, -1, -1, -1,  0,  0,  0,  0,  1,  1,  1, -1])

        stepx = np.concatenate((stepx, _stepx)) * step
        stepy = np.concatenate((stepy, _stepy)) * step

    # Generate single interleaved guider+spectrograph script:
    dith_script = []

    log.debug('{:>5s} {:>5s} {:>5s} {:>5s}'.format('dx', 'dy', 'dRA', 'dDec'))

    for j, (dx, dy) in enumerate(zip(stepx, stepy)):
        ra = ra + dx
        dec = dec + dy

        log.debug('{:5g} {:5g} {:5g} {:5g}'.format(
            dx.to('arcsec').value,
            dy.to('arcsec').value,
            (ra-RA).to('arcsec').value,
            (dec-DEC).to('arcsec').value))

        # Logging variables sent to ICS for output to FITS headers:
        passthru = '{{ OFFSTRA:{:g}, OFFSTDEC:{:g}, TILEID:{:d}, TILERA:{:g}, TILEDEC:{:g} }}'.format((ra-RA).to('arcsec').value, (dec-DEC).to('arcsec').value, tile_id, tile_ra, tile_dec)

        # Start a guide sequence.
        dith_script.append({'sequence'         : 'Guide',
                            'flavor'           : 'science',
                            'exptime'          : 5.0,
                            'action'           : 'start_guiding',
                            'acquisition_time' : 15.0,
                            'deltara'          : dx.to('arcsec').value,
                            'deltadec'         : dy.to('arcsec').value,
                            'correct_for_adc'  : False,
                            'usetemp'          : False,
                            'uselut'           : False,
                            'resetrot'         : False,
                            'passthru'         : passthru,
                            'program'          : 'Dither tile_id {:05d} ({:g} {:g})'.format(tile_id, ra.to('arcsec').value, dec.to('arcsec').value)})

        # Take a spectrograph exposure.
        dith_script.append({'sequence'         : 'Spectrographs',
                            'flavor'           : 'science',
                            'obstype'          : 'SCIENCE',
                            'correct_for_adc'  : False,
                            'usetemp'          : False,
                            'uselut'           : False,
                            'resetrot'         : False,
                            'exptime'          : 60.0,
                            'passthru'         : passthru,
                            'program'          : 'Dither tile_id {:05d} ({:g} {:g})'.format(tile_id, ra.to('arcsec').value, dec.to('arcsec').value)})

        # Break, then manually stop guiding before starting the next exposure.
        dith_script.append({'sequence'         : 'Break'})

    # Dump JSON guider + spectrograph script into one file.
    dith_filename = 'dithseq_tile_{:06d}_step_{}arcsec_dra{}_ddec{}_{}.json'.format(tile_id, step.value, args.deltara, args.deltadec, args.pattern)

    json.dump(dith_script, open(dith_filename, 'w'), indent=4)
    log.info('Use {} in the DESI observer console.'.format(dith_filename))


def setup_fibermode(args):
    """Set up and write out JSON script for a fiber dithering sequence.

    Parameters
    ----------
    args : argparse.Namespace
        Argument list from command line fibermode subprogram.
    """
    # Generate single interleaved DESI sequence + spectrograph script:
    dith_script = []

    minid, maxid = args.tilerange
    tile_ids = range(minid, maxid+1, 1)

    log.debug('{:>7s} {:>7s} {:>7s}'.format('Tile', 'RA', 'Dec'))

    for tile_id in tile_ids:
        tile_ra, tile_dec = get_tile_coords(tile_id, args.dryrun)
        log.debug('{:7d} {:7g} {:7g}'.format(tile_id, tile_ra, tile_dec))

        # Logging variables sent to ICS for output to FITS headers:
        passthru = '{{ TILEID:{:d}, TILERA:{:g}, TILEDEC:{:g} }}'.format(tile_id, tile_ra, tile_dec)

        # DESI sequence.
        dith_script.append({'sequence'            : 'DESI',
                            'flavor'              : 'science',
                            'fiberassign'         : tile_id,
                            'exptime'             : 60.0,
                            'obstype'             : 'SCIENCE',
                            'guider_exptime'      : 5.0,
                            'acquisition_exptime' : 20.0,
                            'program'             : 'Acquisition of Dither Test Field {}'.format(tile_id),
                            'simulatemoves'       : False,
                            'fvc_exptime'         : 2.0})

        # Take a spectrograph exposure.
        dith_script.append({'sequence'         : 'Spectrographs',
                            'flavor'           : 'science',
                            'obstype'          : 'SCIENCE',
                            'correct_for_adc'  : False,
                            'usetemp'          : False,
                            'uselut'           : False,
                            'resetrot'         : False,
                            'exptime'          : 60.0,
                            'passthru'         : passthru,
                            'program'          : 'Dither fibermode tile_id {:d} ({:g}, {:g})'.format(tile_id, tile_ra, tile_dec)})

        # Break, then manually stop guiding before starting the next exposure.
        dith_script.append({'sequence'         : 'Break'})

    # Dump JSON DESI + spectrograph list into a file.
    dith_filename = 'dithseq_fibermode_{:06d}-{:06d}.json'.format(minid, maxid)
    json.dump(dith_script, open(dith_filename, 'w'), indent=4)
    log.info('Use {} in the DESI observer console.'.format(dith_filename))


if __name__ == '__main__':
    p = ArgumentParser(description='Raster/fiber dithering JSON ICS setup program')
    p.add_argument('-d', '--dryrun', action='store_true', default=False,
                   help='Dry run: do not check for fiberassign files')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Verbose output for debugging')

    sp = p.add_subparsers(title='subcommands', description='valid subcommands',
                          help='additional help')

    prmode = sp.add_parser('rastermode', help='Telescope raster program')
    prmode.add_argument('-t', '--tileid', required=True, type=int,
                        help='Tile ID used for raster')
    prmode.add_argument('-s', '--step', required=True, type=float,
                        help='Raster step size in arcsec')
    prmode.add_argument('-p', '--pattern', choices=['3x3','5x5'], default='3x3',
                        help='Raster pattern')
    prmode.add_argument('--plot', action='store_true', default=False,
                        help='Plot raster pattern for debugging')
    prmode.add_argument('--deltara', default=0, type=float,
                        help='RA offset raster center from Tile RA')
    prmode.add_argument('--deltadec', default=0, type=float,
                        help='Dec offset raster center from Tile Dec')
    prmode.set_defaults(func=setup_rastermode)

    pfmode = sp.add_parser('fibermode', help='Fiber dither program')
    pfmode.add_argument('-t', '--tilerange', required=True, nargs=2, type=int,
                        help='Min/max tile ID for positioners')
    pfmode.set_defaults(func=setup_fibermode)

    args = p.parse_args()
    if 'func' in args:
        if args.verbose:
            log.basicConfig(format='%(levelname)s: %(message)s', level=log.DEBUG)
        else:
            log.basicConfig(format='%(levelname)s: %(message)s', level=log.INFO)
        args.func(args)
    else:
        p.print_help()
