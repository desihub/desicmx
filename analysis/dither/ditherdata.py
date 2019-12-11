from configparser import ConfigParser
from glob import glob
import os

from astropy.table import Table
from astropy.io import fits
from astropy.io import ascii

import numpy as np


class DitherSequence:
    """Access class to dithering sequence data from nightwatch or redux
    files."""

    def __init__(self, inifile, dry_run, output):
        """Parse a configuration file in INI format.

        Parameters
        ----------
        inifile : str
            Name of INI file with configuration data.
        dry_run : bool
            If true, do not process input files.
        output : str
            Name of output file (FITS format).
        """

        config = ConfigParser()
        config.read(inifile)
        sequence = config['dithersequence']

        # Set up the output.
        self._output = output

        # Set up the file type and exposure sequence.
        self._location = sequence['location']
        self._filetype = sequence['filetype']
        self._date = sequence['date']
        self._exposures = [int(e) for e in sequence['exposures'].split()]

        if 'coordinates' not in config:
            raise ValueError('no coordinates set for dither!')
        
        coords = config['coordinates']
        self._dithertype = coords['dithertype']
        
        self._wcs = fits.getdata(coords['wcsfile'], 2)
        self._wcs = self._wcs[np.argsort(self._wcs['mjd_obs'])]
        self._central_exposure = int(sequence['centralexposure'])

        if coords['dithertype'] == 'telescope':
            fadir = coords['fiberassigndir']
            self._ditherfa = fits.getdata(os.path.join(
                fadir, 'fiberassign-%s.fits' % coords['ditheredtilenum']))
            self._unditherfa = fits.getdata(os.path.join(
                fadir, 'fiberassign-%s.fits' % coords['unditheredtilenum']))
            expnum = [int(fn.split('-')[1]) for fn in self._wcs['filename']]
            centralind = expnum.index(self._central_exposure)
            self._central_wcs = self._wcs[centralind]

            # Set the Tile ID for the output metadata.
            self._tileid = coords['unditheredtilenum']
        else:
            raise ValueError('not implemented')

        # Extract the list of exposures on disk.
        self._exposure_files = self._getfilenames()

        if not dry_run:
            # Construct fiber output.
            self._exposure_table = self._buildtable()

    def _getfilenames(self):
        """Return a list of exposures and filenames given an INI configuration.

        Returns
        -------
        exfiles : dict
            Dictionary of exposure numbers and corresponding nightwatch
            qcframe or redux sframe FITS files.
        """

        # Set up the path and file prefix depending on the filetype.
        if self._filetype == 'nightwatch':
            fileprefix = 'qcframe'

            if self._location == 'nersc':
                prefix = '/global/project/projectdirs/desi/spectro/nightwatch/kpno'
            elif self._location == 'kpno':
                prefix = '/exposures/desi' # not correct path!
            else:
                raise ValueError('Unknown location {}'.format(self._location))
        elif self._filetype == 'redux':
            fileprefix = 'sframe'

            if self._location == 'nersc':
                prefix = '/global/project/projectdirs/desi/spectro/redux/daily/exposures'
            elif self._location == 'kpno':
                prefix = '/exposures/desi' # not correct path!
            else:
                raise ValueError('Unknown location {}'.format(self._location))
        else:
            raise ValueError('Unknown file type {}'.format(self._filetype))

        # Find the exposures files.
        exfiles = {}
        for ex in self._exposures:
            folder = '{}/{}/{:08d}'.format(prefix, self._date, ex)
            files = sorted(glob('{}/{}*.fits'.format(folder, fileprefix)))
            exfiles[ex] = files

        return exfiles

    def _buildtable(self):
        """Loop through the exposure list and construct an observation
        table."""

        tabrows = []

        for i, (expid, exfiles) in enumerate(self._exposure_files.items()):
            specflux_b, specflux_r, specflux_z = [], [], []
            tab = None

            if len(exfiles) == 0:
                continue

            print(expid)
            for exfile in exfiles:
                print(exfile)
                hdu = fits.open(exfile)

                # The following tables are present in the redux sframes and the
                # nightwatch qcframes.
                wave = hdu['WAVELENGTH'].data

                # However, in the nightwatch files the wavelength data are a
                # table of size nfiber x nwavelength.
                if self._filetype == 'nightwatch':
                    if wave.ndim > 1:
                        wave = wave[0]

                fluxhead = hdu['FLUX'].header
                fluxdata = hdu['FLUX'].data
                ivardata = hdu['IVAR'].data
                fibermap = hdu['FIBERMAP'].data
                exptime = fluxhead['EXPTIME']
                if not np.all(self._unditherfa['FIBER'] ==
                              np.arange(len(self._unditherfa))):
                    raise ValueError('weird fiberassign file format!')
                fibermap = self._unditherfa[fibermap['FIBER']]

                target_id = fibermap['TARGETID']
                target_ra = fibermap['TARGET_RA']
                target_dec = fibermap['TARGET_DEC']
                fiber = fibermap['FIBER']
                objtype = fibermap['OBJTYPE']
                flux_g = fibermap['FLUX_G']
                flux_r = fibermap['FLUX_R']
                flux_z = fibermap['FLUX_Z']
                x, y = [fibermap['FIBERASSIGN_{}'.format(val)] for val in ('X', 'Y')]

                camera = fluxhead['CAMERA'][0].upper()

                if getattr(self, '_deltara', None) is not None:
                    dra  = self._deltara[i]*np.ones(len(fiber))
                    ddec = self._deltadec[i]*np.ones(len(fiber))
                elif self._dithertype == 'telescope':
                    dithra = self._ditherfa['target_ra']
                    dithdec = self._ditherfa['target_dec']
                    udithra = self._unditherfa['target_ra']
                    udithdec = self._unditherfa['target_dec']
                    ontarget = ((self._ditherfa['targetid'] ==
                                 self._unditherfa['targetid']) &
                                (self._ditherfa['objtype'] == 'TGT'))
                    dfiberra = (dithra-udithra)*np.cos(np.radians(udithdec))*60*60
                    dfiberdec = (dithdec-udithdec)*60*60
                    if not np.all(self._ditherfa['FIBER'] ==
                                  np.arange(len(self._ditherfa))):
                        raise ValueError('unexpected shape of dither file')
                    dfiberra[~ontarget] = np.nan
                    dfiberdec[~ontarget] = np.nan
                    dfiberra = dfiberra[fiber]
                    dfiberdec = dfiberdec[fiber]
                    wcs = self.lookup_wcs(fluxhead['MJD-OBS'])
                    centralwcs = self._central_wcs
                    if (~np.isfinite(centralwcs['cenra'][1]) or
                        ~np.isfinite(centralwcs['cendec'][1])):
                        raise ValueError('central pointing ra/dec is NaN!')
                    dtelra = (wcs['cenra'][1]-centralwcs['cenra'][1])
                    dtelra *= np.cos(np.radians(centralwcs['cendec'][1]))
                    dteldec = wcs['cendec'][1]-centralwcs['cendec'][1]
                    dra = dfiberra + dtelra*60*60
                    ddec = dfiberdec + dteldec*60*60
                    if np.all(~np.isfinite(dra)):
                        print('warning: no good telescope offset for %s' %
                              exfile)
                else:
                    raise ValueError('not implemented')
                    
                for j, fiber_id in enumerate(fiber):
                    flux = fluxdata[j]
                    ivar = ivardata[j]
                    if not np.any(ivar > 0):
                        specflux = 0
                        specflux_ivar = 0
                    else:
                        meanivar = np.mean(ivar[ivar > 0])
                        mask = ivar > meanivar / 100
                        specflux = np.trapz(flux*mask, wave)
                        specflux_ivar = 1./np.sum(ivar[mask]**-1)
                        # Schlegel: sum over correct wavelengths, all three
                        # filters, plus 11 pixel median filter to reject
                        # cosmics.
                        # will require being better about reading in
                        # the spectrographs together.
                    tabrows.append((expid, exptime,
                                    target_id[j], target_ra[j], target_dec[j],
                                    fiber[j], objtype[j],
                                    flux_g[j], flux_r[j], flux_z[j],
                                    specflux, specflux_ivar, camera,
                                    dra[j], ddec[j],
                                    x[j], y[j]))

        tab = Table(rows=tabrows,
                    names=('EXPID', 'EXPTIME',
                           'TARGETID', 'TARGET_RA', 'TARGET_DEC',
                           'FIBER', 'OBJTYPE',
                           'FLUX_G', 'FLUX_R', 'FLUX_Z',
                           'SPECTROFLUX', 'SPECTROFLUX_IVAR', 'CAMERA',
                           'DELTA_X_ARCSEC', 'DELTA_Y_ARCSEC',
                           'XFOCAL', 'YFOCAL'),
                    meta={'EXTNAME' : 'DITHER',
                          'TILEID' : '{}'.format(self._tileid)})

        return tab

    def lookup_wcs(self, mjd):
        # expfn = self._exposure_files[expnum]
        # mjd = fits.getheader(expfn)['MJD-OBS']
        ind = np.searchsorted(self._wcs['mjd_obs'], mjd)
        if ind >= len(self._wcs):
            return np.array(((np.nan,)*3, (np.nan,)*3),
                            dtype=[('cenra', '3f8'), ('cendec', '3f8')])
        twcs = self._wcs[ind]
        if twcs['mjd_obs'] <= mjd:
            raise ValueError('Something confusing with wcs list')
        return twcs

    def save(self, filename=None, overwrite=True):
        """Save exposure table to a FITS file.

        Parameters
        ----------
        filename : str
            Output filename. If none, use default output class member.
        overwrite : bool
            If true, clobber an existing file with the same name.
        """
        if filename is None:
            filename = self._output
        self._exposure_table.write(filename, overwrite=overwrite)

    def rearrange_table(self):
        exps = np.sort(np.unique(self._exposure_table['EXPID']))
        nexp = len(exps)
        nfiber = 5000
        camera = np.unique(self._exposure_table['CAMERA'])
        out = {}
        newtab = np.zeros((nfiber, nexp), dtype=[
            ('expid', 'i4'), ('exptime', 'f4'),
            ('targetid', 'i8'), ('camera', 'U1'),
            ('target_ra', 'f8'), ('target_dec', 'f8'),
            ('fiber', 'i4'), ('objtype', 'U3'),
            ('flux_g', 'f4'), ('flux_r', 'f4'), ('flux_z', 'f4'),
            ('spectroflux', 'f4'),
            ('spectroflux_ivar', 'f4'),
            ('delta_x_arcsec', 'f4'), ('delta_y_arcsec', 'f4'),
            ('xfocal', 'f4'), ('yfocal', 'f4')])
        newtab['fiber'][:] = -999
        for camera0 in camera:
            mc = self._exposure_table['CAMERA'] == camera0
            newtab0 = newtab.copy()
            newtab0['camera'] = camera0
            for i, exp0 in enumerate(exps):
                me = self._exposure_table['EXPID'] == exp0
                dat = self._exposure_table[me & mc]
                if np.any(dat['FIBER'] >= 5000) or np.any(dat['FIBER'] < 0):
                    raise ValueError('unexpected number of fibers')
                ind = dat['FIBER']
                for field in dat.dtype.names:
                    newtab0[field.lower()][ind, i] = dat[field]
            out[camera0] = newtab0
        return out

    def __str__(self):
        """String representation of the exposure sequence.
        """
        output = ['Tile ID {}'.format(self._tileid)]
        for ex, files in self._exposure_files.items():
            filenames = '- exposure {:08d}\n'.format(ex)
            for f in files:
                filenames = '{}  + {}\n'.format(filenames, f)
            output.append(filenames)

        return '\n'.join(output)

