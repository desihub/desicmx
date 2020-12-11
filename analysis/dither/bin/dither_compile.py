#!/usr/bin/env python

import os
import glob
import ditherdata


if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser(
        description='Compile dither data from exposures')
    parse.add_argument('-n', '--night', type=int, help='YYYYMMDD of EXPID')
    parse.add_argument('-e', '--expid', type=int, nargs='+',
                       help='EXPID of exposures composing the sequence')
    parse.add_argument('-o', '--output', help='file name of output')
    parse.add_argument('-l', '--location', default='nersc',
                       help='location of files; nersc or kpno')
    parse.add_argument('-r', '--reduction', default='redux',
                       help='type of reduction; redux or nightwatch')
    args = parse.parse_args()
    expfn = ditherdata.getfilenames(args.expid, args.night, args.reduction,
                                    args.location)
    rawdir = ('/global/cfs/cdirs/desi/spectro/data'
              if args.location == 'nersc'
              else '/data/dts/exposures')  # can't remember.
    unditherfapath = os.path.join(rawdir, f'{args.night:8d}',
                                  f'{args.expid[0]:08d}')
    unditherfafn = glob.glob(os.path.join(
        unditherfapath, 'fiberassign*.fits*'))
    if len(unditherfafn) != 1:
        raise ValueError('could not find undithered fiberassign file in '
                         'unditherfapath.')
    unditherfafn = unditherfafn[0]
    table = ditherdata.buildtable(expfn, args.reduction, 'fiber',
                                  rawdir=rawdir, unditherfa=unditherfafn)
    rearranged = ditherdata.rearrange_table(table)
    ditherdata.write_table(rearranged, args.output)
