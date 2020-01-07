import os, argparse
import numpy as np

import fitsio
from astropy.table import Table

#import desimodel.io
#import matplotlib.pyplot as plt

datadir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'data')
reduxdir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'redux', 'daily', 'exposures')
nightwatchdir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'nightwatch', 'kpno')
outdir = os.path.join(os.getenv('DESI_ROOT'), 'ioannis', 'cmx', 'fluxcalib')

def read_and_stack_nightwatch(night, verbose=False, overwrite=False):
    """Read and stack all the nightwatch QA files for a given night.

    """
    from glob import glob
    import astropy.table
    #import desimodel.io

    stackwatchfile = os.path.join(outdir, 'qa-nightwatch-{}.fits'.format(str(night)))
    if os.path.isfile(stackwatchfile) and not overwrite:
        print('Reading {}'.format(stackwatchfile))
        allqa = Table(fitsio.read(stackwatchfile))
    else:
        #print('Reading the focal plane model.')
        #fp = desimodel.io.load_focalplane()[0]
        #fp = fp['PETAL', 'FIBER', 'OFFSET_X', 'OFFSET_Y']

        nightdir = os.path.join(nightwatchdir, night)
        allexpiddir = glob(os.path.join(nightdir, '????????'))

        # Need to assign fiberassign files algorithmically...
        #tilefile = os.path.join(datadir, night, os.path.basename(exp)
        #                         '00037738', 'fiberassign-063511.fits')
        #tile = Table(fitsio.read(fiberfile))

        allqa = []
        for expiddir in allexpiddir:
            expid = os.path.basename(expiddir)
            qafile = os.path.join(expiddir, 'qa-{}.fits'.format(expid))

            # Hack!
            requestfile = os.path.join(datadir, night, 'request-{}.fits'.format(expid))
            tileid = int(jj['PASSTHRU'].split(':')[3].split(',')[0])
            tilefile = glob(os.path.join(datadir, night, '????????', 'fiberassign-{:06d}.fits'.format(tileid)))[0]
            #print(fitsio.FITS(tilefile)['FIBERASSIGN'])
            if verbose:
                print('Reading {}'.format(tilefile))
            tile = Table(fitsio.FITS(tilefile)['FIBERASSIGN'].read())

            if verbose:
                print('Reading {}'.format(qafile))
            qa = Table(fitsio.read(qafile, 'PER_CAMFIBER'))

            # Add the fiberassign info - hack!
            qa = astropy.table.join(qa, tile, keys='FIBER')

            allqa.append(qa)
        allqa = astropy.table.vstack(allqa)

            # Need to update the data model to 'f4'.
            if False:
                print('Updating the data model.')
                for col in allqa.colnames:
                    if allqa[col].dtype == '>f8':
                        allqa[col] = allqa[col].astype('f4')

            # Add the x/y offsets from the focal plane model
            allqa = astropy.table.join(allqa, fp, keys='FIBER')

        print('Writing {}'.format(stackwatchfile))
        fitsio.write(stackwatchfile, allqa.as_array(), clobber=True)

    return allqa

def main():

    parser = argparse.ArgumentParser(description='Derive flux-calibrated spectra and estimate the throughput.')

    parser.add_argument('-n','--night', type=int, default=20200102, required=True, help='night')
    args = parser.parse_args()



    # In[261]:


    allqa = read_and_stack_nightwatch(allnight, verbose=True, overwrite=True)
    allqa


    # #### Find fibers with high S/N spectra.

    # In[262]:


    def select_spectra_snr(data, thiscam='Z', snrcut=10, qaplot=False):
        """Select spectra based on S/N."""

        #wcam = np.where(data['CAM'] == thiscam)[0]
        #wgood = np.where((data['MEDIAN_CALIB_SNR'] > snrcut) * (data['CAM'] == thiscam))[0]
        wsnr = np.where((data['MEDIAN_CALIB_SNR'] > snrcut))[0]

        isnr = np.where(np.isin(data['FIBER'], data['FIBER'][wsnr]))[0]
        print('Found {} high S/N spectra in {}/{} unique fibers.'.format(
            len(isnr), len(set(data['FIBER'][isnr])), len(set(data['FIBER']))))

        #for fib in set(star['FIBER']):
        #    if np.sum(star['FIBER'] == fib) != 3:
        #        print('Problem with fiber {}'.format(fib))    

        if qaplot:
            col = iter(colors)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(data['OFFSET_X'], data['OFFSET_Y'], 'k.',
                    ms=1, alpha=0.5, label='')
            for petal in sorted(set(data['PETAL'][isnr])):
                ii = data['PETAL'][isnr] == petal
                ax.scatter(data['OFFSET_X'][isnr][ii], data['OFFSET_Y'][isnr][ii], 
                           marker='o', s=20, color=next(col), label='Petal {}'.format(petal))
            ax.legend(frameon=True, fontsize=12, ncol=1)
            ax.axis('off')

        return data[isnr]


    # In[263]:


    star = select_spectra_snr(allqa, snrcut=10, qaplot=True)
    star
    #star[star['FIBER'] == 3525]
    #star[star['FIBER'] == 7]


    # In[300]:


    def read_one_spectrocam(night, expid, spectrocam, fibers=None, verbose=False):
        """Read a single-camera spectrum for a given night, expid, 
        and spectrograph+camera combination.

        """
        import desispec.io

        strnight = str(night)
        strexpid = '{:08d}'.format(expid)

        specfile = os.path.join(reduxdir, strnight, strexpid, 'sframe-{}-{}.fits'.format(
            spectrocam, strexpid))
        if verbose:
            print('Reading {}'.format(specfile))

        spec = desispec.io.read_frame(specfile)
        if fibers is not None:
            #print('Selecting {} fibers'.format(len(fibers)))
            wfiber = np.where(np.isin(spec.fibers, fibers))[0]
            spec.flux = spec.flux[wfiber, :]
            spec.ivar = spec.ivar[wfiber, :]
            spec.mask = spec.mask[wfiber, :]
            spec.resolution_data = spec.resolution_data[wfiber, :, :]
            spec.fibermap = spec.fibermap[wfiber]
            spec.fibers = fibers

        return spec

    def read_all_spectra(data, verbose=False):
        """Pack all the spectra for a given night and expid into a single 
        dictionary.

        """
        from astropy.table import Table, Column

        # Night and expid better be unique!
        night, expid = data['NIGHT'][0], data['EXPID'][0]

        # Build an output dictionary with one key per fiber.
        fibers, uindx = np.unique(star['FIBER'].data, return_index=True)
        #fibers = sorted(set(data['FIBER']))
        nspec = len(fibers)
        print('Reading {} spectra.'.format(nspec))

        out = Table()    
        #out.add_column(Column(name='FIBER', data=np.array(fibers).reshape(1, nspec)))
        #out.add_column(Column(name='OFFSET_X', data=data['OFFSET_X'][uindx].reshape(1, nspec)))
        #out.add_column(Column(name='OFFSET_Y', data=data['OFFSET_Y'][uindx].reshape(1, nspec)))
        out['FIBER'] = fibers
        out['OFFSET_X'] = data['OFFSET_X'][uindx].data
        out['OFFSET_Y'] = data['OFFSET_Y'][uindx].data
        out['PETAL'] = data['PETAL'][uindx].data
        out['TARGET_RA'] = data['TARGET_RA'][uindx].data
        out['TARGET_DEC'] = data['TARGET_DEC'][uindx].data

        waveout = Table()

        offset = 0
        for ispectro, spectro in enumerate(sorted(set(data['SPECTRO']))):
            wspectro = np.where(data['SPECTRO'] == spectro)[0]

            # I think this assumes that all cameras are operational.
            thesefibers = np.array(sorted(set(data['FIBER'][wspectro])))

            for icam, cam in enumerate(sorted(set(data['CAM'][wspectro]))):
                spectrocam = '{}{:d}'.format(cam.lower(), spectro)
                #print(spectrocam, offset, len(thesefibers))

                onespec = read_one_spectrocam(night, expid, spectrocam, 
                                              fibers=thesefibers, verbose=verbose)
                if ispectro == 0:
                    waveout['WAVE_{}'.format(cam)] = onespec.wave
                    here!
                    out['FLUX_{}'.format(cam)] = np.zeros((nspec, onespec.nwave), dtype='f4')
                out['FLUX_{}'.format(cam)][offset:offset+len(thesefibers), :] = onespec.flux

            offset = offset + len(thesefibers)

        return waveout, out

    def plot_all_spectra(allspec, nplot=None):
        """Plot the data given the output of read_all_spectra.

        """
        from IPython.display import Image

        if nplot is None:
            nplot = len(allspec['FIBER'])

        for ifib, fib in enumerate(allspec['FIBER'][:nplot]):
            fig, ax = plt.subplots(figsize=(6, 4))
            #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            #for cam in ('B', 'R', 'Z'):
            #    ax.plot(allspec['WAVE_{}'.format(cam)], allspec['FLUX_{}'.format(cam)][ifib, :])

            url = "http://legacysurvey.org/viewer-dev/jpeg-cutout?ra={:.6f}&dec={:.6f}&layer=dr8&size=100".format(
                allspec['TARGET_RA'][ifib], allspec['TARGET_DEC'][ifib])
            print(url)
            Image(url=url)
        #ax.plot(bcam.wave, bthru.photons(bcam.wave, bcam.flux[fiberid, :]))


    # In[301]:


    #jj = read_one_spectrocam(20200102, 37742, 'b0', fibers=[7, 10, 144])
    #jj.flux.shape


    # In[302]:


    get_ipython().run_line_magic('time', 'allspec = read_all_spectra(star, verbose=False)')
    print(allspec.keys())


    # In[292]:


    from astropy.table import Table, Column
    bb = Table()
    bb.add_column(Column(name='bob', data=np.array([1, 2]).reshape(1, 2)))#, shape=(1, 2), length=1))
    bb


if __name__ == '__main__':
    main()
