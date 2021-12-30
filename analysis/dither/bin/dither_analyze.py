#!/usr/bin/env python

import os
import glob
import ditherdata
import solvedither


if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser(
        description='Analyze dither data')
    parse.add_argument('data', help='name of dither data fits file to analyze')
    parse.add_argument('-o', '--outdir', default='.', help='output directory')
    parse.add_argument('-l', '--label', default=None,
                       help='label for file names')
    parse.add_argument('-n', '--nthreads', default=0, type=int,
                       help='number of threads to use')
    parse.add_argument('-p', '--petals',
                       type=int, nargs='+',
                       default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parse.add_argument('-x', '--extra-psfparam-guess', default=None,
                       type=float, nargs='+')
    args = parse.parse_args()
    data = ditherdata.read_table(args.data)
    if args.label is not None:
        label = args.label
    else:
        label = (os.path.join(*(os.path.basename(args.data).split('.')[:-1])) +
                 '-%s')
    solvedither.process_all(data, outdir=args.outdir, label=label,
                            threads=args.nthreads, usepetals=args.petals,
                            extra_psfparam_guess=args.extra_psfparam_guess,
                            useguess=False)
