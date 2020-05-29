#!/usr/bin/env python3

import argparse
import astropy.io.fits as fits
from astropy.table import Table
import glob
import numpy as np
import os

extnames = ['GUIDE0', 'FOCUS1', 'GUIDE2', 'GUIDE3', 'FOCUS4', 'GUIDE5', 'FOCUS6', 'GUIDE7', 'GUIDE8', 'FOCUS9']

def check_quadrants(image):

    q1 = image[516:1032, 1174:2198]
    q2 = image[516:1032, 50:1074]
    q3 = image[0:516, 50:1074]
    q4 = image[0:516, 1174:2198]

    result = {1: np.sum(q1 != 0) == 0,
              2: np.sum(q2 != 0) == 0,
              3: np.sum(q3 != 0) == 0,
              4: np.sum(q4 != 0) == 0}

    return result

def check_valid_extname(extname):
    if not extname in extnames:
        print('invalid extension name specified')
        assert(False)

def _expid_from_fname(fname):
    _fname = os.path.split(fname)[1]
    expid = int(_fname[4:12])

    return expid

def _get_file_list(night, basedir, min_expid, max_expid):

    assert(min_expid <= max_expid)

    flist = glob.glob(basedir + '/' + night + '/*/gfa*.fz')
 
    flist.sort()

    _flist = [os.path.split(f)[1] for f in flist]
    
    expid = np.array([_expid_from_fname(f) for f in _flist])

    flist = np.array(flist)

    flist = flist[(expid >= min_expid) & (expid <= max_expid)]

    return flist

if __name__ == "__main__":
    descr = 'check GFA images for missing/empty quadrants'
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

    args = parser.parse_args()

    min_expid = args.min_expid
    max_expid = args.max_expid

    if args.expid is not None:
        min_expid = args.expid
        max_expid = args.expid

    flist = _get_file_list(args.night[0], args.basedir, min_expid, max_expid)

    if args.extname is not None:
        _extnames = [args.extname]
        for e in _extnames: check_valid_extname(e)
    else:
        _extnames = extnames

    results = []
    for i, fname in enumerate(flist):
        for extname in _extnames:
            print('working on : ', fname, extname)
            # try/except handles case where not all GFA cameras present in an exposure
            try:
                im = fits.getdata(fname, extname=extname)
            except:
                continue
            result = check_quadrants(im)
            result['fname'] = fname
            result['extname'] = extname
            results.append(result)

    amps = ['G', 'H', 'E', 'F']
    for r in results:
        for k in np.arange(1, 5):
            if r[k] == 1:
                print(r['fname'], r['extname'], ': amp ', amps[k-1], ' is empty')
