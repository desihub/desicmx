#!/usr/bin/env python3

import argparse
import astropy.io.fits as fits
from astropy.table import Table
import glob
import numpy as np
import os

extnames = ['GUIDE0', 'FOCUS1', 'GUIDE2', 'GUIDE3', 'FOCUS4', 'GUIDE5', 'FOCUS6', 'GUIDE7', 'GUIDE8', 'FOCUS9']

def fname_to_expid():
    print('blat')

def n_fake_bad(im, thresh):
    # calculate number of bad pixels in overscan/prescan regions for a single-camera image
    n = np.sum(im[:, 0:50] > thresh) + np.sum(im[:, 1074:1174] > thresh) + np.sum(im[:, 2198:2248] > thresh)

    return n

def check_valid_extname(extname):
    if not extname in extnames:
        print('invalid extension name specified')
        assert(False)

def _get_file_list(night, basedir, min_expid, max_expid):

    assert(min_expid <= max_expid)

    flist = glob.glob(basedir + '/' + night + '/*/gfa*.fz')
 
    _flist = [os.path.split(f)[1] for f in flist]
    
    expid = np.array([int(f[4:12]) for f in _flist])

    flist = np.array(flist)

    flist = flist[(expid >= min_expid) & (expid <= max_expid)]

    return flist

def _expid_from_fname(fname):
    _fname = os.path.split(fname)[1]
    expid = int(_fname[4:12])

    return expid

if __name__ == "__main__":
    descr = 'print information about GFA prescan/overscan bad pixels'
    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument('night', type=str, nargs=1, 
                        help='NIGHT string')

    parser.add_argument('--basedir', default='/exposures/desi', type=str,
                        help='raw exposure base directory')

    parser.add_argument('--min_expid', default=-1, type=int,
                        help='minimum EXPID')

    parser.add_argument('--max_expid', default=10000000, type=int,
                        help='maximum EXPID')

    parser.add_argument('--expid', default=None, type=int,
                        help='run just exposure EXPID')

    parser.add_argument('--extname', default=None, type=str,
                        help='only analyze EXTNAME')

    parser.add_argument('--thresh', default=10000, type=int,
                        help='pixel value threshold for badness')

    parser.add_argument('--outname', default=None,
                        help='output file name for results summary file')

    args = parser.parse_args()


    outname = args.outname

    min_expid = args.min_expid
    max_expid = args.max_expid

    if args.expid is not None:
        min_expid = args.expid
        max_expid = args.expid

    flist = _get_file_list(args.night[0], args.basedir, min_expid, max_expid)

    if args.extname is not None:
        _extnames = [args.extname]
    else:
        _extnames = extnames

    print('filename', 'extname', 'npix_bad')
    print('====================================')

    result = []
    for i, fname in enumerate(flist):
        for extname in _extnames:
            im = fits.getdata(fname, extname=extname)
            nbad = n_fake_bad(im, args.thresh)
            print(fname, extname, nbad)
            result.append((args.night[0], fname, _expid_from_fname(fname), extname, nbad))
        if i != (len(flist)-1):
            print('-')

    if args.outname is not None:
        assert(not os.path.exists(outname))
        night = [t[0] for t in result]
        fname = [t[1] for t in result]
        expid = [t[2] for t in result]
        extname = [t[3] for t in result]
        npix_bad = [t[4] for t in result]

        t = Table()
        t['NIGHT'] = night
        t['FNAME'] = fname
        t['EXPID'] = expid
        t['EXTNAME'] = extname
        t['NPIX_BAD'] = npix_bad

        t.write(outname, format='fits')
