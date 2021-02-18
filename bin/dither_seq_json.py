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

def get_tile_info(tileid, dryrun=False):
    """Open fiberassign header and read tile info.

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
    isdith : bool
        True if fiber positions have been dithered.
    """

    if dryrun:
        return 0., 0., True

    if 'DOS_DESI_TILES' in os.environ:
        path = os.environ['DOS_DESI_TILES']
    else:
        # Fallback to hardcoded path (issue a warning).
        path = '/data/tiles/ALL_tiles/20191119'
        log.warning('DOS_DESI_TILES undefined; using {}'.format(path))

    # Warning (SB): subfolder in DOS_DESI_TILES likely to change.
    fibfile = '/'.join([path, '080', 'fiberassign-{:06d}.fits.gz'.format(tileid)])
    try:
        hdus = fits.open(fibfile)
    except FileNotFoundError as e:
        log.error('TILE {} cannot be found.\n{}.'.format(tileid, e))
        raise SystemExit

    header = hdus['PRIMARY'].header
    ra, dec = header['TILERA'], header['TILEDEC']

    try:
        isdith = header['ISDITH']
    except KeyError as e:
        print(e)
        print(fibfile)
        print('Use ISDITH=True')
        isdith = True

    return ra, dec, isdith


def setup_rastermode(args):
    """Set up and write out JSON telescope raster script for dither tests.

    Parameters
    ----------
    args : argparse.Namespace
        Argument list from command line rastermode subprogram.
    """

    RA = args.deltara*u.arcsec
    DEC = args.deltadec*u.arcsec
    ra = RA
    dec = DEC

    # Set step size. Add a unit from astropy, does not change value.
    step = args.step*u.arcsec
    tile_id = args.tileid
    tile_ra, tile_dec, tile_dith = get_tile_info(tile_id, args.dryrun)

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
                            'guider_exptime'   : 5.0,
                            'action'           : 'start_guiding',
                            'acquisition_time' : 15.0,
                            'deltara'          : dx.to('arcsec').value,
                            'deltadec'         : dy.to('arcsec').value,
#                            'correct_for_adc'  : False,
#                            'usetemp'          : False,
#                            'uselut'           : False,
#                            'resetrot'         : False,
                            'passthru'         : passthru,
                            'program'          : 'Dither tile_id {:05d} ({:g} {:g})'.format(tile_id, ra.to('arcsec').value, dec.to('arcsec').value)})

        # Take a spectrograph exposure.
        dith_script.append({'sequence'         : 'Spectrographs',
                            'flavor'           : 'science',
                            'obstype'          : 'SCIENCE',
#                            'correct_for_adc'  : False,
#                            'usetemp'          : False,
#                            'uselut'           : False,
#                            'resetrot'         : False,
                            'exptime'          : 90.0,
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

    for i, tile_id in enumerate(tile_ids):
        tile_ra, tile_dec, tile_dith = get_tile_info(tile_id, args.dryrun)
        if args.dryrun:
            if i == 0:
                tile_dith = False

        log.debug('{:7d} {:7g} {:7g}'.format(tile_id, tile_ra, tile_dec))

        # Logging variables sent to ICS for output to FITS headers:
        passthru = '{{ TILEID:{:d}, TILERA:{:g}, TILEDEC:{:g} }}'.format(tile_id, tile_ra, tile_dec)

        # Program name. Note that first tile/exposure is undithered.
        if tile_dith:
            prog = 'Dither fibermode tile {:d} ({:g}, {:g})'.format(tile_id, tile_ra, tile_dec)
        else:
            prog = 'Dither fibermode undithered tile {:d} ({:g}, {:g})'.format(tile_id, tile_ra, tile_dec)

        # Stack up DESI sequences. Note: exptime is for spectrographs.
        dith_script.append({'sequence'            : 'DESI',
                            'fiberassign'         : tile_id,
                            'exptime'             : args.exptime,
                            'guider_exptime'      : 5.0,
                            'acquisition_exptime' : 15.0,
                            'fvc_exptime'         : 2.0,
                            'sky_exptime'         : args.skyexptime,
                            'focus_exptime'       : args.focusexptime,
                            'movedelay'           : args.movedelay,
                            'program'             : prog})
        # add 1 min pause for cool down
        if args.pause > 0:
            if tile_id != maxid:
                dith_script.append({'sequence'            : 'Action',
                                    'action'              : 'pause',
                                    'required'            : ['SPECTRO', 'CCDS'],
                                    'delay'               : args.pause})

#        if i > 0:
#            dith_script[-1]['correct_for_adc'] = False
#
#        # Break needed to manually stop guiding?
#        dith_script.append({'sequence'         : 'Break'})

    # Dump JSON DESI + spectrograph list into a file.
    if args.pause > 0:
        dith_filename = 'dithseq_fibermode_{:06d}_{:06d}_pause.json'.format(minid, maxid)
    else:
        dith_filename = 'dithseq_fibermode_{:06d}_{:06d}.json'.format(minid, maxid)
    json.dump(dith_script, open(dith_filename, 'w'), indent=4)
    log.info('Use {} in the DESI observer console.'.format(dith_filename))


if __name__ == '__main__':
    # Main options specified before rastermode and fibermode subcommands.
    p = ArgumentParser(description='Raster/fiber dithering JSON ICS setup program',
                       formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-d', '--dryrun', action='store_true', default=False,
                   help='Dry run: do not check for fiberassign files')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Verbose output for debugging')

    sp = p.add_subparsers(title='subcommands', description='valid subcommands',
                          help='additional help')

    # Raster mode: move telescope boresight with deltara/deltadec
    prmode = sp.add_parser('rastermode', formatter_class=ArgumentDefaultsHelpFormatter,
                          help='Telescope raster program ("lost in space")')
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

    # Fiber mode: set up a sequence of fiberassign tiles.
    pfmode = sp.add_parser('fibermode', formatter_class=ArgumentDefaultsHelpFormatter,
                           help='Fiber dither program ("precision dither")')
    pfmode.add_argument('-t', '--tilerange', required=True, nargs=2, type=int,
                        help='Min/max tile ID for positioners')
    pfmode.add_argument('-e', '--exptime', type=float, default=180.0,
                        help='Exposure time [seconds]')
    pfmode.add_argument('-f', '--focusexptime', type=float, default=60.0,
                        help='Focus loop time [seconds]')
    pfmode.add_argument('-m', '--movedelay', type=float, default=0.0,
                        help='Move delay [seconds]')
    pfmode.add_argument('-s', '--skyexptime', type=float, default=60.0,
                        help='Sky loop time [seconds]')
    pfmode.add_argument('-p', '--pause', type=float, default=0.0,
                        help='Pause for cooldown [seconds]')
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
