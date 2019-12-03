#!/usr/bin/env python
"""Generate GFA and spectrograph ICS sequences for running telescope dither
tests. Assumes a 3x3 dither pattern.

The user can specify the step size, the name of the tile_id, and the central
coordinates of the tile_id (not used at the moment).

One can specify an offset of the raster center from the current location;
the script will return one to the current location at the end.  And the 
labels on the raster positions will be relative to the current location. 
This is so that one can do multiple rasters in the same coordinate system.

Based on instructions available at

https://desi.lbl.gov/trac/wiki/CommissioningCommissioningPlanning/procedures#GAIAstardithertests
"""

from argparse import ArgumentParser
import json
import numpy as np
from astropy import units as u

parser = ArgumentParser(description='Dither GFA+Spectrograph scripter')

# Dithering options for file headers.
parser.add_argument('step', type=float, help='Dither step size [arcsec]')
parser.add_argument('-t', '--tile_id', default=0, type=int,
                    help='Tile ID from fiberassign')
parser.add_argument('-r', '--deltara', default=0, type=float,
                    help='Ra offset of center of raster from current position')
parser.add_argument('-d', '--deltadec', default=0, type=float,
                    help='Dec offset of center of raster from current position')

args = parser.parse_args()

RA = args.deltara*u.arcsec
DEC = args.deltadec*u.arcsec
ra = RA
dec = DEC

step = args.step*u.arcsec # Add a unit from astropy, does not change value
tile_id = args.tile_id

stepx = np.asarray([ 0,  1, -1, -1,  0,  1,  1,  0, -1, -1,  1])*step
stepy = np.asarray([ 0,  1,  0,  0, -1,  0,  0, -1,  0,  0,  1])*step

gfa_filename = 'gfaseq_tile_id{:05d}_{}arcsec_dra{}_ddec{}.json'.format(tile_id, step.value, args.deltara, args.deltadec)
spec_filename = 'specseq_tile_id{:05d}_{}arcsec_dra{}_ddec{}.json'.format(tile_id, step.value, args.deltara, args.deltadec)

gfa_script = []
spec_script = []

gfa_script.append({'sequence': 'Action',
                   'action'  : 'slew',
                   'deltara' : RA.value,
                   'deltadec': DEC.value
                  })

for j, (dx, dy) in enumerate(zip(stepx, stepy)):
    ra = ra + dx
    dec = dec + dy

    # GFA sequence:
    if j > 0:
        gfa_script.append({'sequence': 'Break'})
        gfa_script.append({'sequence': 'Action',
                           'action'  : 'slew',
                           'deltara' : dx.value,
                           'deltadec': dy.value
                          })

    gfa_script.append({'sequence': 'GFA',
                       'flavor'  : 'science',
                       'exptime' : 30.0,
                       'count'   : 2,
                       'correct_for_adc' : False,
                       'program' : "Dither tile_id {:05d} ({:g},{:g})".format( tile_id, 
                                                                                  (ra).to('arcsec').value,
                                                                                  (dec).to('arcsec').value
                                                                                )
                      })

    # Spectrograph sequence:
    spec_script.append({'sequence' : 'Spectrographs',
                        'flavor'   : 'science',
                        'count'    : 1,
                        'exptime'  : 60.0,
                        'program'  : "Dither tile_id {:05d} ({:g},{:g})".format( tile_id, 
                                                                                    (ra).to('arcsec').value,
                                                                                    (dec).to('arcsec').value),
                        'obstype'  : "SCIENCE"
                       })
    if j+1 < len(stepx):
        spec_script.append({'sequence' : 'Break'})

gfa_script.append({'sequence': 'Action',
                   'action'  : 'slew',
                   'deltara' : -RA.value,
                   'deltadec': -DEC.value
                  })

json.dump(gfa_script, open(gfa_filename, 'w'), indent=4)
json.dump(spec_script, open(spec_filename, 'w'), indent=4)

print('Use {} in the DESI observer console.'.format(gfa_filename))
print('Use {} in the spectrograph ICS console.'.format(spec_filename))
