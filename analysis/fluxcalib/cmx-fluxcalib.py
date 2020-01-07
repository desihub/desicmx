import os, argparse, pdb
import numpy as np

import fitsio
from glob import glob
import desispec.io

#import desimodel.io
#import matplotlib.pyplot as plt

datadir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'data')
reduxdir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'redux', 'daily', 'exposures')
nightwatchdir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'nightwatch', 'kpno')
outdir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'redux', 'ioannis', 'cmx', 'fluxcalib')
#outdir = os.path.join(os.getenv('DESI_ROOT'), 'ioannis', 'cmx', 'fluxcalib')

def read_and_stack_nightwatch(night, verbose=False, overwrite=False):
    """Read and stack all the nightwatch QA files for a given night.

    """
    import json
    import astropy.table
    from astropy.table import Table
    #import desimodel.io

    stackwatchfile = os.path.join(outdir, 'qa-nightwatch-{}.fits'.format(str(night)))
    fiberassignmapfile = os.path.join(outdir, 'fiberassignmap-{}.fits'.format(str(night)))

    if os.path.isfile(stackwatchfile) and not overwrite:
        print('Reading {}'.format(stackwatchfile))
        data = Table(fitsio.read(stackwatchfile, 'PER_CAMFIBER'))
        fiberassignmap = Table(fitsio.read(fiberassignmapfile))
    else:
        #print('Reading the focal plane model.')
        #fp = desimodel.io.load_focalplane()[0]
        #fp = fp['PETAL', 'FIBER', 'OFFSET_X', 'OFFSET_Y']

        nightdir = os.path.join(nightwatchdir, str(night))
        allexpiddir = glob(os.path.join(nightdir, '????????'))

        data = []
        fiberassignmap = Table(names=('NIGHT', 'EXPID', 'TILEID', 'FIBERASSIGNFILE'),
                               dtype=('U8', 'U8', 'i4', 'U32'))
        for expiddir in allexpiddir:
            expid = os.path.basename(expiddir)
            qafile = os.path.join(expiddir, 'qa-{}.fits'.format(expid))

            qaFITS = fitsio.FITS(qafile)
            if 'PER_CAMFIBER' in qaFITS:
                if verbose:
                    print('Reading {}'.format(qafile))
                qa = Table(qaFITS['PER_CAMFIBER'].read())

            # Hack! Figure out the mapping between EXPID and FIBERMAPusing the request-EXPID.json file.
            requestfile = os.path.join(datadir, str(night), expid, 'request-{}.json'.format(expid))
            with open(requestfile) as ff:
                req = json.load(ff)
            if 'PASSTHRU' in req.keys():
                tileid = int(req['PASSTHRU'].split(':')[3].split(',')[0])
                tilefile = glob(os.path.join(datadir, str(night), '????????', 'fiberassign-{:06d}.fits'.format(tileid)))
                #if len(tilefile) == 0:
                #    print('No fibermap file found for EXPID={}'.format(expid))
                #if len(tilefile) > 0:
                #    print('Multiple fibermap files found for EXPID={}!'.format(expid))
                if len(tilefile) > 0:
                    tsplit = tilefile[0].split('/')
                    fiberassignmap.add_row((str(night), str(expid), tileid, os.path.join(tsplit[-2], tsplit[-1])))
                    #fiberassignmap[str(expid)] = [tileid]
                    data.append(qa)
        data = astropy.table.vstack(data)

        # Need to update the data model to 'f4'.
        print('Updating the data model.')
        for col in data.colnames:
            if data[col].dtype == '>f8':
                data[col] = data[col].astype('f4')

        print('Writing {}'.format(stackwatchfile))
        fitsio.write(stackwatchfile, data.as_array(), clobber=True, extname='PER_CAMFIBER')

        print('Writing {}'.format(fiberassignmapfile))
        # ValueError: unsupported type 'U42'
        #fitsio.write(fiberassignmapfile, fiberassignmap.as_array(), clobber=True)
        fiberassignmap.write(fiberassignmapfile, overwrite=True)

    return data, fiberassignmap

def select_stdstars(data, fiberassignmap, snrcut=10, verbose=False):
    """Select spectra based on S/N and targeting bit.

    """
    from astropy.table import Table
    from desitarget.targets import main_cmx_or_sv

    night, expid = data['NIGHT'][0], data['EXPID'][0]

    tileid = fiberassignmap['TILEID'][0]
    fibermapfile = fiberassignmap['FIBERASSIGNFILE'][0]
    fibermapfile = os.path.join(datadir, str(night), fibermapfile)
    if verbose:
        print('Reading {}'.format(fibermapfile))
    fibermap = Table(fitsio.read(fibermapfile))#, columns=['CMX_TARGET']))

    # Pre-select standards
    target_colnames, target_masks, survey = main_cmx_or_sv(fibermap)
    target_col, target_mask = target_colnames[0], target_masks[0] # CMX_TARGET, CMX_MASK for commissioning
    istd = np.where((fibermap[target_col] & target_mask.mask('STD_BRIGHT') != 0))[0]
    # For each standard, apply a minimum S/N cut (in any camera) to
    # subselect the fibers that were reasonably centered on the fiber.
    reject = np.ones(len(istd), dtype=bool)
    if len(istd) > 0:
        for ii, fiber in enumerate(fibermap['FIBER'][istd]):
            wspec = np.where((data['EXPID'] == expid) * (data['FIBER'] == fiber))[0]
            if len(wspec) > 0:
                isnr = np.where(data['MEDIAN_CALIB_SNR'][wspec] > snrcut)[0]
                if len(isnr) > 0:
                    reject[ii] = False
    if np.sum(reject) > 0:
        istd_reject = istd[reject]
    else:
        istd_reject = np.array([])
    if verbose:
        print('EXPID={}, TILEID={}, nstd={}/{}'.format(
            expid, tileid, len(istd)-len(istd_reject), len(istd), snrcut))

    # Hack! Set the targeting bit for the failed standards to zero!
    if len(istd_reject) > 0:
        fibermap[target_col][istd_reject] = 0

    return fibermap
    
def main():

    parser = argparse.ArgumentParser(description='Derive flux-calibrated spectra and estimate the throughput.')

    parser.add_argument('-n', '--night', type=int, default=20200102, required=True, help='night')
    parser.add_argument('--snrcut', type=np.float32, default=10.0, help='S/N cut.')
    parser.add_argument('--verbose', action='store_true', help='Be verbose.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    args = parser.parse_args()

    data, fiberassignmap = read_and_stack_nightwatch(args.night, verbose=args.verbose,
                                                     overwrite=args.overwrite)

    # For each EXPID, find the fibers with standard stars *and* good spectra.
    for expid in set(data['EXPID']):
        thismap = (fiberassignmap['EXPID'] == '{:08d}'.format(expid)) * (fiberassignmap['NIGHT'] == str(args.night))
        thisspec = (data['EXPID'] == expid) * (data['NIGHT'] == args.night)
        
        fullfibermap = select_stdstars(data[thisspec], fiberassignmap[thismap],
                                       snrcut=args.snrcut, verbose=args.verbose)
        
        #from desitarget.cmx.cmx_targetmask import cmx_mask  
        #print(np.sum(fibermap['CMX_TARGET'] & cmx_mask.mask('STD_BRIGHT') != 0))

        strexpid = '{:08d}'.format(expid)
        framefiles = glob(os.path.join(reduxdir, str(args.night), strexpid, 'frame-[brz][0-9]-{}.fits'.format(strexpid)))
        for framefile in framefiles:
            outframefile = os.path.join(outdir, str(args.night), strexpid, os.path.basename(framefile))
            if os.path.isfile(outframefile) and not args.overwrite:
                print('File exists {}'.format(outframefile))
                continue

            # Read and copy all the columns!
            if args.verbose:
                print('Reading {}'.format(framefile))
            fr = desispec.io.read_frame(framefile)
            fibermap = fullfibermap[np.isin(fullfibermap['FIBER'], fr.fibermap['FIBER'])]
            for col in fibermap.dtype.names:
                if col in fr.fibermap.dtype.names:
                    fr.fibermap[col] = fibermap[col]

            if args.verbose:
                print('  Writing {}'.format(outframefile))
            desispec.io.write_frame(outframefile, fr)




if __name__ == '__main__':
    main()
