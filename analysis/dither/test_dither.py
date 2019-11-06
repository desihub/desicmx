from ditherdata import DitherSequence
from argparse import ArgumentParser

parser = ArgumentParser(description='Dither exposure analysis')
parser.add_argument('-c', '--config', dest='config', default='dither.ini',
                    help='Configuration file (INI format).')
parser.add_argument('-o', '--output', dest='output', default=None,
                    help='FITS output name.')
args = parser.parse_args()

ditherseq = DitherSequence(args.config)
if args.output is not None:
    ditherseq.save(args.output)

#print(ditherseq._exposures)
#print(ditherseq._exposure_files)
#print(ditherseq)
