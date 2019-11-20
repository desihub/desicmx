from ditherdata import DitherSequence
from argparse import ArgumentParser

parser = ArgumentParser(description='Dither exposure analysis')
parser.add_argument('-c', '--config', dest='inifile', default='dither.ini',
                    help='Configuration file (INI format).')
parser.add_argument('-d', '--dry-run', dest='dry_run', action='store_true',
                    help='Test without processing data.')
parser.add_argument('-o', '--output', dest='output', default=None,
                    help='FITS output name.')
args = parser.parse_args()

ditherseq = DitherSequence(**vars(args))

if args.dry_run:
    print(ditherseq)
else:
    if args.output is not None:
        ditherseq.save(args.output)
