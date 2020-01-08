import os, argparse, pdb
import numpy as np
from glob import glob

import fitsio
import desispec.io
from desispec.calibfinder import CalibFinder
from desispec.fluxcalibration import isStdStar

datadir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'data')
reduxdir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'redux', 'daily', 'exposures')
nightwatchdir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'nightwatch', 'kpno')
outdir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'redux', 'ioannis', 'cmx', 'fluxcalib')

def gather_qa(night, verbose=False, overwrite=False):
    """Read and stack all the nightwatch QA files for a given night.

    """
    import json
    import astropy.table
    from astropy.table import Table

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
    istd = np.where(isStdStar(fibermap))[0]
    target_colnames, target_masks, survey = main_cmx_or_sv(fibermap)
    target_col, target_mask = target_colnames[0], target_masks[0] # CMX_TARGET, CMX_MASK for commissioning
    #istd = np.where((fibermap[target_col] & target_mask.mask('STD_BRIGHT') != 0))[0]
    
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
                    
    if np.sum(~reject) > 0:
        istd_keep = istd[~reject]
    else:
        istd_keep = []
    if np.sum(reject) > 0:
        istd_reject = istd[reject]
    else:
        istd_reject = []
        
    # Hack! Set the targeting bit for the failed standards to zero!
    if len(istd_reject) > 0:
        fibermap[target_col][istd_reject] = 0

    if verbose:
        print('EXPID={}, TILEID={}, NSTD={}/{}'.format(
            expid, tileid, len(istd_keep), len(istd), snrcut))

    return fibermap

def update_frame_fibermaps(data, fiberassignmap, verbose=False, overwrite=False):
    """Update the fibermaps in each frame file with the necessary information on
    targeted standard stars.

    """
    from desitarget.targets import main_cmx_or_sv
    
    night = data['NIGHT'][0]
    
    # For each EXPID, find the fibers with standard stars *and* good spectra.
    for expid in set(data['EXPID']):
        thismap = (fiberassignmap['EXPID'] == '{:08d}'.format(expid)) * (fiberassignmap['NIGHT'] == str(night))
        thisspec = (data['EXPID'] == expid) * (data['NIGHT'] == night)
        
        fullfibermap = select_stdstars(data[thisspec], fiberassignmap[thismap], verbose=verbose)
        
        #from desitarget.cmx.cmx_targetmask import cmx_mask  
        #print(np.sum(fibermap['CMX_TARGET'] & cmx_mask.mask('STD_BRIGHT') != 0))

        strexpid = '{:08d}'.format(expid)
        framefiles = glob(os.path.join(reduxdir, str(night), strexpid, 'frame-[brz][0-9]-{}.fits'.format(strexpid)))
        for framefile in framefiles:
            outframefile = os.path.join(outdir, 'exposures', str(night), strexpid, os.path.basename(framefile))
            if os.path.isfile(outframefile) and not overwrite:
                print('File exists {}'.format(outframefile))
                continue

            # Read and copy all the columns!
            #if verbose:
            #    print('Reading {}'.format(framefile))
            fr = desispec.io.read_frame(framefile)
            fibermap = fullfibermap[np.isin(fullfibermap['FIBER'], fr.fibermap['FIBER'])]

            # Require at least one standard star on this petal.
            istd = np.where(isStdStar(fibermap))[0]
            #target_colnames, target_masks, survey = main_cmx_or_sv(fibermap)
            #target_col, target_mask = target_colnames[0], target_masks[0] # CMX_TARGET, CMX_MASK for commissioning
            #istd = np.where((fibermap[target_col] & target_mask.mask('STD_BRIGHT') != 0))[0]
            if len(istd) > 0:
                # Fragile. desi_proc should be smarter about how it initializes
                # the dummy fibermap and/or we should be smarter about what
                # columns we assume that desi_proc has added...
                fibermap['OBJTYPE'] = fr.fibermap['OBJTYPE'] # fragile!
                fibermap['DESI_TARGET'] |= fr.fibermap['DESI_TARGET'] # fragile!
                fr.fibermap = fibermap

                if verbose:
                    print('Writing {}'.format(outframefile))
                desispec.io.write_frame(outframefile, fr)

        print('Returning after first EXPID - fix me.')
        return

def fit_stdstars(night, verbose=False, overwrite=False):
    """Fit the standards on each petal.

    desi_fit_stdstars --frames 00028833/frame-*.fits --skymodels 00028833/sky-*.fits \
      --fiberflats $DESI_SPECTRO_CALIB/spec/sp3/fiberflat-sm4-*-20191108.fits \
      --starmodels $DESI_BASIS_TEMPLATES/stdstar_templates_v2.2.fits --outfile stdstars-3-00028833.fits

    """
    starmodelfile = os.path.join(os.getenv('DESI_BASIS_TEMPLATES'), 'stdstar_templates_v2.2.fits')

    allexpiddir = glob(os.path.join(outdir, 'exposures', str(night), '????????'))
    for expiddir in allexpiddir:
        expid = os.path.basename(expiddir)
        # Process each spectrograph separately.
        for spectro in np.arange(10):
            outstdfile = os.path.join(expiddir, 'stdstars-{}-{}.fits'.format(str(spectro), expid))
            if os.path.isfile(outstdfile) and not overwrite:
                print('File exists {}'.format(outstdfile))
                continue

            framefiles = sorted(glob(os.path.join(expiddir, 'frame-[brz]{}-{}.fits'.format(str(spectro), expid))))
            # Gather the calibration files.
            if len(framefiles) == 3:
                skymodelfiles = [framefile.replace(outdir, reduxdir).replace('frame-', 'sky-') for framefile in framefiles]
                fiberflatfiles = []
                for framefile in framefiles:
                    calib = CalibFinder([fitsio.read_header(framefile)])
                    fiberflatfiles.append(os.path.join(os.getenv('DESI_SPECTRO_CALIB'), calib.data['FIBERFLAT']))

                cmd = 'desi_fit_stdstars --frames {framefiles} --skymodels {skymodelfiles} --fiberflats {fiberflatfiles} '
                cmd += '--starmodels {starmodelfile} --outfile {outstdfile}'
                cmd = cmd.format(framefiles=' '.join(framefiles),
                                 skymodelfiles=' '.join(skymodelfiles),
                                 fiberflatfiles=' '.join(fiberflatfiles),
                                 starmodelfile=starmodelfile,
                                 outstdfile=outstdfile)
                #print(cmd)
                os.system(cmd)

def fluxcalib(night, verbose=False, overwrite=False):
    """Perform flux-calibration.

    desi_compute_fluxcalibration  --infile 00028833/frame-b3-*.fits --sky 00028833/sky-b3-*.fits \
      --fiberflat $DESI_SPECTRO_CALIB/spec/sp3/fiberflat-sm4-b-20191108.fits --models stdstars-3-00028833.fits \
      --outfile fluxcalib-b3-00028833.fits --delta-color-cut 12

    """
    allexpiddir = glob(os.path.join(outdir, 'exposures', str(night), '????????'))
    for expiddir in allexpiddir:
        expid = os.path.basename(expiddir)
        
        # Process each spectrograph separately.
        for spectro in np.arange(10):
            modelsfile = os.path.join(expiddir, 'stdstars-{}-{}.fits'.format(str(spectro), expid))
            if os.path.isfile(modelsfile):
                framefiles = sorted(glob(os.path.join(expiddir, 'frame-[brz]{}-{}.fits'.format(str(spectro), expid))))
                for framefile in framefiles:
                    outcalibfile = framefile.replace('frame-', 'fluxcalib-')
                    if os.path.isfile(outcalibfile) and not overwrite:
                        print('File exists {}'.format(outcalibfile))
                        continue
                    
                    # Gather the calibration files.
                    skyfile = framefile.replace(outdir, reduxdir).replace('frame-', 'sky-')
                    calib = CalibFinder([fitsio.read_header(framefile)])
                    fiberflatfile = os.path.join(os.getenv('DESI_SPECTRO_CALIB'), calib.data['FIBERFLAT'])

                    cmd = 'desi_compute_fluxcalibration --infile {framefile} --sky {skyfile} '
                    cmd += '--fiberflat {fiberflatfile} --models {modelsfile} --outfile {outcalibfile} '
                    cmd += '--delta-color-cut 12'
                    cmd = cmd.format(framefile=framefile, skyfile=skyfile,
                                     fiberflatfile=fiberflatfile, modelsfile=modelsfile,
                                     outcalibfile=outcalibfile)
                    #print(cmd)
                    os.system(cmd)

def process_exposures(night, verbose=False, overwrite=False):
    """Process all exposures and write out cFrame files.

    usage: desi_process_exposure [-h] -i INFILE [--fiberflat FIBERFLAT]
                                 [--sky SKY] [--calib CALIB] -o OUTFILE
                                 [--cosmics-nsig COSMICS_NSIG]
                                 [--sky-throughput-correction]
    Apply fiberflat, sky subtraction and calibration.
    optional arguments:
      -h, --help            show this help message and exit
      -i INFILE, --infile INFILE
                            path of DESI exposure frame fits file
      --fiberflat FIBERFLAT
                            path of DESI fiberflat fits file
      --sky SKY             path of DESI sky fits file
      --calib CALIB         path of DESI calibration fits file
      -o OUTFILE, --outfile OUTFILE
                            path of DESI sky fits file
      --cosmics-nsig COSMICS_NSIG
                            n sigma rejection for cosmics in 1D (default, no
                            rejection)
      --sky-throughput-correction
                            apply a throughput correction when subtraction the sky

    """
    allexpiddir = glob(os.path.join(outdir, 'exposures', str(night), '????????'))
    for expiddir in allexpiddir:
        expid = os.path.basename(expiddir)

        framefiles = sorted(glob(os.path.join(expiddir, 'frame-[brz]?-{}.fits'.format(expid))))
        for framefile in framefiles:
            outframefile = framefile.replace('frame-', 'cframe-')
            if os.path.isfile(outframefile) and not overwrite:
                print('File exists {}'.format(outframefile))
                continue

            # Gather the calibration files.
            skyfile = framefile.replace(outdir, reduxdir).replace('frame-', 'sky-')
            calib = CalibFinder([fitsio.read_header(framefile)])
            fiberflatfile = os.path.join(os.getenv('DESI_SPECTRO_CALIB'), calib.data['FIBERFLAT'])
            calibfile = framefile.replace('frame-', 'fluxcalib-')

            cmd = 'desi_process_exposure --infile {framefile} --sky {skyfile} '
            cmd += '--fiberflat {fiberflatfile} --calib {calibfile} '
            cmd += '--cosmics-nsig 6 --outfile {outframefile} '
            cmd = cmd.format(framefile=framefile, skyfile=skyfile,
                             fiberflatfile=fiberflatfile, calibfile=calibfile,
                             outframefile=outframefile)
            #print(cmd)
            os.system(cmd)

def qaplots(night, verbose=False, overwrite=False):
    """Make some QAplots.

    cal is electrons/A  /  (ergs/s/cm2/A)
    
    ./plot_fluxcalib.py fluxcalib-*-00029181.fits \
      /project/projectdirs/desi/datachallenge/reference_runs/19.9/spectro/redux/mini/exposures/20200411/00000041/calib-*4-00000041.fits

    """
    import astropy.units as u
    from astropy import constants as const
    import matplotlib.pyplot as plt

    import desimodel.io
    
    pngfile = os.path.join(outdir, 'qa-throughtput-{}.png'.format(str(night)))
    if os.path.isfile(pngfile) and not overwrite:
        print('File exists {}'.format(pngfile))
        return

    desi = desimodel.io.load_desiparams()
    area = (desi['area']['geometric_area'] * u.m**2).to(u.cm**2)

    thru = dict()
    for camera in ('b', 'r' , 'z'):
        thru[camera] = desimodel.io.load_throughput(camera)
        #import specter.throughput
        #thrufile = '/global/u2/i/ioannis/repos/desihub/desimodel/data/throughput/thru-{}.fits'.format(camera)
        #thru[camera] = specter.throughput.load_throughput(thrufile)

    fig, ax = plt.subplots()
    
    allexpiddir = glob(os.path.join(outdir, 'exposures', str(night), '????????'))
    for expiddir in allexpiddir:
        expid = os.path.basename(expiddir)

        for spectro in np.arange(10):
            calibfile = glob(os.path.join(expiddir, 'fluxcalib-r{}-{}.fits'.format(spectro, expid)))
            if len(calibfile) > 0:
                for camera in ('b', 'r' , 'z'):
                    cframefile = os.path.join(expiddir, 'cframe-{}{}-{}.fits'.format(camera, spectro, expid))
                    calibfile = os.path.join(expiddir, 'fluxcalib-{}{}-{}.fits'.format(camera, spectro, expid))

                    info = fitsio.FITS(calibfile)
                    hdr = info['FLUXCALIB'].read_header()
                    exptime = hdr['EXPTIME'] * u.s

                    wave = info['WAVELENGTH'].read() * u.angstrom
                    flux = info['FLUXCALIB'].read()
                    ivar = info['IVAR'].read()
                    
                    specthru = flux * 1e17 * (u.electron / u.angstrom) / (u.erg / u.s / u.cm / u.cm / u.angstrom)
                    specthru *= const.h.to(u.erg * u.s) * const.c.to(u.angstrom / u.s) / exptime / area / wave # [electron/photon]

                    # Plot just the standards--
                    fibermap = desispec.io.read_fibermap(cframefile)
                    istd = np.where(isStdStar(fibermap))[0]

                    for ii in istd:
                        label = '{}-{}-{}'.format(int(expid), str(spectro), fibermap['FIBER'][ii])
                        ax.plot(wave.value, specthru[ii, :].value, label=label)
     
    for camera in ('b', 'r' , 'z'):
        ax.plot(wave.value, thru[camera].thru(wave.value), color='grey')
                    
    ax.set_xlabel(r'Wavelength ($\AA$)')
    ax.set_ylabel('Throughput (electron / photon)')
    ax.set_ylim(0, 1)
    ax.legend()

    print('Writing {}'.format(pngfile))
    fig.savefig(pngfile)
               
def main():

    parser = argparse.ArgumentParser(description='Derive flux-calibrated spectra and estimate the throughput.')

    parser.add_argument('-n', '--night', type=int, default=20200102, required=True, help='night')
    parser.add_argument('--gather-qa', action='store_true', help='Gather and stack nightwatch QA files.')
    parser.add_argument('--update-fibermaps', action='store_true', help='Update fibermap files')
    parser.add_argument('--fit-stdstars', action='store_true', help='Fit the standard stars.')
    parser.add_argument('--fluxcalib', action='store_true', help='Do the flux-calibration.')
    parser.add_argument('--process-exposures', action='store_true', help='Process all exposures.')
    parser.add_argument('--group-spectra', action='store_true', help='Group the data into healpixels.')
    parser.add_argument('--redrock', action='store_true', help='Do redshift fitting.')

    parser.add_argument('--qaplots', action='store_true', help='Make some plots.')
    parser.add_argument('--verbose', action='store_true', help='Be verbose.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    args = parser.parse_args()

    # Gather QA files
    if args.gather_qa:
        data, fiberassignmap = gather_qa(args.night, verbose=args.verbose, overwrite=args.overwrite)

    # Update fibermap and frame files.
    if args.update_fibermaps:
        data, fiberassignmap = gather_qa(args.night, verbose=args.verbose)
        update_frame_fibermaps(data, fiberassignmap, verbose=args.verbose, overwrite=args.overwrite)

    # Fit the standard stars.
    if args.fit_stdstars:
        fit_stdstars(args.night, verbose=args.verbose, overwrite=args.overwrite)

    if args.fluxcalib:
        fluxcalib(args.night, verbose=args.verbose, overwrite=args.overwrite)

    if args.process_exposures:
        process_exposures(args.night, verbose=args.verbose, overwrite=args.overwrite)

    if args.group_spectra:
        spectradir = os.path.join(outdir, 'spectra-64')
        cmd = 'desi_group_spectra --reduxdir {outdir} --nights {night} --outdir {spectradir}'
        cmd = cmd.format(outdir=outdir, night=str(args.night), spectradir=spectradir)
        os.system(cmd)

    if args.redrock:
        spectrafiles = glob(os.path.join(outdir, 'spectra-64', 'spectra-64-*.fits'))
        for spectrafile in spectrafiles:
            zbestfile = os.path.join(outdir, 'spectra-64', os.path.basename(spectrafile).replace('spectra-', 'zbest-'))
            redrockfile = os.path.join(outdir, 'spectra-64', os.path.basename(spectrafile).replace('spectra-', 'redrock-').replace('.fits', '.h5'))
            cmd = 'rrdesi --output {redrockfile} --zbest {zbestfile} {spectrafiles} --mp 32'
            cmd = cmd.format(redrockfile=redrockfile, zbestfile=zbestfile, spectrafiles=' '.join(spectrafiles))
            os.system(cmd)
            
        pdb.set_trace()
        
    if args.qaplots:
        qaplots(args.night, verbose=args.verbose, overwrite=args.overwrite)

if __name__ == '__main__':
    main()
