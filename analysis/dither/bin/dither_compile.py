#!/usr/bin/env python

import os
import glob
import ditherdata


if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser(
        description='Compile dither data from exposures')
    parse.add_argument('-n', '--night', type=int, help='YYYYMMDD of EXPID')
    parse.add_argument('-e', '--expid', type=str, nargs='+',
                       help='EXPID of exposures composing the sequence')
    parse.add_argument('-o', '--output', help='file name of output',
                       required=True)
    parse.add_argument('-l', '--location', default='nersc',
                       help='location of files; nersc or kpno')
    parse.add_argument('-r', '--reduction', default='redux',
                       help='type of reduction; redux or nightwatch')
    parse.add_argument('-d', '--desimeterdir', default=None,
                       help='directory to find desimeter reductions')
    args = parse.parse_args()
    expidstr = args.expid
    expid = []
    for expid0 in expidstr:
        if '-' in expid0:
            try:
                first, last = expid0.split('-')
            except Exception:
                raise ValueError('weird expid string: ', expid0)
            expid = expid + list(range(int(first), int(last)+1))
        else:
            try:
                expid.append(int(expid0))
            except Exception:
                raise ValueError('weird expid string: ', expid0)
    expfn = ditherdata.getfilenames(expid, args.night, args.reduction,
                                    args.location)
    rawdir = ('/global/cfs/cdirs/desi/spectro/data'
              if args.location == 'nersc'
              else '/data/dts/exposures/raw')  # can't remember.
    unditherfapath = os.path.join(rawdir, f'{args.night:8d}',
                                  f'{expid[0]:08d}')
    unditherfafn = glob.glob(os.path.join(
        unditherfapath, 'fiberassign*.fits*'))
    if len(unditherfafn) != 1:
        raise ValueError('could not find undithered fiberassign file in '
                         'unditherfapath.')
    unditherfafn = unditherfafn[0]
    dithertype = 'fiber' if args.desimeterdir is None else 'desimeter'
    table = ditherdata.buildtable(expfn, args.reduction, dithertype,
                                  rawdir=rawdir, unditherfa=unditherfafn,
                                  desimeterdir=args.desimeterdir)
    rearranged = ditherdata.rearrange_table(table)
    ditherdata.write_table(rearranged, args.output)
