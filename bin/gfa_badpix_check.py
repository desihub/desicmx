#!/usr/bin/env python3

import argparse
import astropy.io.fits as fits
from astropy.table import Table
import glob
import numpy as np
import os

extnames = ['GUIDE0', 'FOCUS1', 'GUIDE2', 'GUIDE3', 'FOCUS4', 'GUIDE5', 'FOCUS6', 'GUIDE7', 'GUIDE8', 'FOCUS9']

def n_fake_bad(im, thresh):
    # calculate number of bad pixels in overscan/prescan regions for a single-camera image
    n = np.sum(im[:, 0:50] > thresh) + np.sum(im[:, 1074:1174] > thresh) + np.sum(im[:, 2198:2248] > thresh)

    return n

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
 
    _flist = [os.path.split(f)[1] for f in flist]
    
    expid = np.array([_expid_from_fname(f) for f in _flist])

    flist = np.array(flist)

    flist = flist[(expid >= min_expid) & (expid <= max_expid)]

    return flist

def _process_pixel_data(data, thresh):
    # data could be either a 2D image or 3D guider image cube
    sh = data.shape
    if len(sh) == 2:
        nbad = [n_fake_bad(data, thresh)]
    else:
        nbad = [n_fake_bad(data[i, :, :], thresh) for i in range(sh[0])]

    return nbad

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
    print('='*35)

    result = []
    for i, fname in enumerate(flist):
        for extname in _extnames:
            # try/except handles case where not all GFA cameras present in an exposure
            try:
                data = fits.getdata(fname, extname=extname)
            except:
                continue
            nbad = _process_pixel_data(data, args.thresh)
            this_result = [(args.night[0], fname, _expid_from_fname(fname), extname, nbad[frame], frame) for frame in range(len(nbad))]
            print(fname, extname, nbad[0])
            result += this_result
        if i != (len(flist)-1):
            print('-')

    if args.outname is not None:
        assert(not os.path.exists(outname))

        t = Table()
        t['NIGHT'] = [t[0] for t in result]
        t['FNAME'] = [t[1] for t in result]
        t['EXPID'] = [t[2] for t in result]
        t['EXTNAME'] = [t[3] for t in result]
        t['NPIX_BAD'] = [t[4] for t in result]
        t['FRAME'] = [t[5] for t in result]
        t['THRESH_ADU'] = args.thresh

        t.write(outname, format='fits')
