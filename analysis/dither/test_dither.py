from ditherdata import DitherSequence
from argparse import ArgumentParser

parser = ArgumentParser(description='Dither exposure analysis')
parser.add_argument('-c', '--config', dest='config', default='dither.ini',
                    help='Configuration file (INI format).')
args = parser.parse_args()

ditherseq = DitherSequence(args.config)
#print(ditherseq._exposures)
#print(ditherseq._exposure_files)
#print(ditherseq)
