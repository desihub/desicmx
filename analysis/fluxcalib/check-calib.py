#!/usr/bin/env python

import os, argparse, pdb
import numpy as np
from glob import glob
import fitsio

import desispec.io
from desispec.calibfinder import CalibFinder

reduxdir = os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv('SPECPROD'), 'exposures')

#nights = desispec.io.get_nights()
nights = sorted([os.path.basename(nn) for nn in glob(os.path.join(reduxdir, '2020????'))])

def nights_and_expid():
    badflats = []
    for night in nights:
        expids = desispec.io.get_exposures(night)
        for expid in expids:
            for spectro in np.arange(10):
                for camera in ('b', 'r', 'z'):
                    framefile = desispec.io.findfile('frame', night=int(night), expid=expid, camera='{}{}'.format(camera, spectro))
                    if not os.path.isfile(framefile):
                        print('Missing {}'.format(framefile))
                        continue
                    hdr = fitsio.read_header(framefile)
                    calib = CalibFinder([hdr])
                    #fr = desispec.io.read_frame(framefile)
                    #calib = CalibFinder([fr.meta])
                    fiberflatfile = os.path.join(os.getenv('DESI_SPECTRO_CALIB'), calib.data['FIBERFLAT'])
                    #print(fiberflatfile)
                    fiberflat = desispec.io.fiberflat.read_fiberflat(fiberflatfile)
                    if np.sum(np.sum(fiberflat.ivar == 0, axis=1) == hdr['NAXIS1']):
                        badflats.append(fiberflatfile)
        pdb.set_trace()

fiberflatfiles = sorted(glob(os.path.join(os.getenv('DESI_SPECTRO_CALIB'), 'spec', 'sm*', 'fiberflatnight-sm*-[brz]-????????.fits')))
for ii, fiberflatfile in enumerate(fiberflatfiles):
    fiberflat = desispec.io.fiberflat.read_fiberflat(fiberflatfile)
    nmasked = len(np.where(np.sum(fiberflat.ivar == 0, axis=1) == fiberflat.nwave)[0])
    print(os.path.basename(fiberflatfile), nmasked)
