from configparser import ConfigParser
from glob import glob
import os

from astropy.table import Table
from astropy.io import fits
from astropy.io import ascii

import numpy as np
import scipy.ndimage


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
        self._wcsind = coords['wcsind'] if 'wcsind' in coords else 'after'
        self._usewcspair = int(coords['usepair']) if 'usepair' in coords else 1
        
        self._wcs = fits.getdata(coords['wcsfile'], 2)
        self._wcs = self._wcs[np.argsort(self._wcs['mjd_obs'])]
        self._central_exposure = int(sequence['centralexposure'])

        if coords['dithertype'] == 'telescope':
            fadir = coords['fiberassigndir']
            self._ditherfa = fits.getdata(os.path.join(
                fadir, 'fiberassign-%s.fits' % coords['ditheredtilenum']))
            self._unditherfa = fits.getdata(os.path.join(
                fadir, 'fiberassign-%s.fits' % coords['unditheredtilenum']))
            expnum = [int(fn.split('.')[0].split('-')[1])
                      for fn in self._wcs['filename']]
            centralind = expnum.index(self._central_exposure)
            self._central_wcs = self._wcs[centralind]

            # Set the Tile ID for the output metadata.
            self._tileid = coords['unditheredtilenum']
        else:
            raise ValueError('not implemented')

        # Extract the list of exposures on disk.
        self._exposure_files = getfilenames(self._exposures, self._date,
                                            self._filetype,
                                            self._location)

        if not dry_run:
            # Construct fiber output.
            self._exposure_table = self._buildtable()


    def _buildtable(self):
        return buildtable(self._exposure_files, self._filetype,
                          self._dithertype,
                          unditherfa=self._unditherfa,
                          ditherfa=self._ditherfa,
                          centralwcs=self._central_wcs,
                          lookup_wcs=self.lookup_wcs,
                          tileid=self._tileid,
                          usewcspair=self._usewcspair)
                   
    def lookup_wcs(self, mjd):
        # expfn = self._exposure_files[expnum]
        # mjd = fits.getheader(expfn)['MJD-OBS']
        ind = np.searchsorted(self._wcs['mjd_obs'], mjd)
        if self._wcsind == 'before':
            ind -= 1
        if ind >= len(self._wcs):
            return np.array(((np.nan,)*3, (np.nan,)*3),
                            dtype=[('cenra', '3f8'), ('cendec', '3f8')])
        twcs = self._wcs[ind]
        if (twcs['mjd_obs'] <= mjd) & (self._wcsind != 'before'):
            raise ValueError('Something confusing with wcs list')
        return twcs

    def rearrange_table(self):
        return rearrange_table(self._exposures_table)

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


def rearrange_table(table):
    exps = np.sort(np.unique(table['EXPID']))
    nexp = len(exps)
    nfiber = 5000
    camera = np.unique(table['CAMERA'])
    out = {}
    newtab = np.zeros((nfiber, nexp), dtype=[
        ('expid', 'i4'), ('exptime', 'f4'), ('mjd_obs', 'f8'),
        ('targetid', 'i8'), ('camera', 'U1'),
        ('target_ra', 'f8'), ('target_dec', 'f8'),
        ('fiber', 'i4'), ('objtype', 'U3'),
        ('flux_g', 'f4'), ('flux_r', 'f4'), ('flux_z', 'f4'),
        ('spectroflux', 'f4'),
        ('spectroflux_ivar', 'f4'),
        ('delta_x_arcsec', 'f4'), ('delta_y_arcsec', 'f4'),
        ('xfocal', 'f4'), ('yfocal', 'f4')])
    newtab['fiber'][:] = -999
    newtab['spectroflux_ivar'][:] = 0
    for camera0 in camera:
        mc = table['CAMERA'] == camera0
        newtab0 = newtab.copy()
        newtab0['camera'] = camera0
        for i, exp0 in enumerate(exps):
            me = table['EXPID'] == exp0
            dat = table[me & mc]
            if np.any(dat['FIBER'] >= 5000) or np.any(dat['FIBER'] < 0):
                raise ValueError('unexpected number of fibers')
            ind = dat['FIBER']
            for field in dat.dtype.names:
                newtab0[field.lower()][ind, i] = dat[field]
        out[camera0] = newtab0
    return out


def buildtable(exposure_files, filetype, dithertype,
               unditherfa=None, ditherfa=None,
               centralwcs=None, lookup_wcs=None, tileid=0,
               verbose=1, usewcspair=1):
    """Loop through the exposure list and construct an observation
    table."""

    tabrows = []
    if ditherfa is not None and isinstance(ditherfa, str):
        ditherfa = fits.getdata(ditherfa, 'FIBERASSIGN')
    if unditherfa is not None and isinstance(unditherfa, str):
        unditherfa = fits.getdata(unditherfa, 'FIBERASSIGN')
        

    for i, (expid, exfiles) in enumerate(exposure_files.items()):
        specflux_b, specflux_r, specflux_z = [], [], []
        tab = None

        if len(exfiles) == 0:
            continue

        if verbose >= 1:
            print(expid)
        if (verbose >= 1) and (lookup_wcs is None or centralwcs is None):
            print('Ignoring any intentional telescope offset...')
        for exfile in exfiles:
            if verbose >= 2:
                print(exfile)
            hdu = fits.open(exfile)

            # The following tables are present in the redux sframes and the
            # nightwatch qcframes.
            wave = hdu['WAVELENGTH'].data

            # However, in the nightwatch files the wavelength data are a
            # table of size nfiber x nwavelength.
            if filetype == 'nightwatch':
                if wave.ndim > 1:
                    wave = wave[0]

            fluxhead = hdu['FLUX'].header
            fluxdata = hdu['FLUX'].data
            # try to zap cosmics
            fluxdata = scipy.ndimage.median_filter(fluxdata, [1, 11])
            ivardata = hdu['IVAR'].data
            fibermap = hdu['FIBERMAP'].data
            exptime = fluxhead['EXPTIME']
            if not np.all(unditherfa['FIBER'] ==
                          np.arange(len(unditherfa))):
                raise ValueError('weird fiberassign file format!')
            fibermap = unditherfa[fibermap['FIBER']]

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

            if dithertype == 'telescope':
                dithra = ditherfa['target_ra']
                dithdec = ditherfa['target_dec']
                udithra = unditherfa['target_ra']
                udithdec = unditherfa['target_dec']
                ontarget = ((ditherfa['targetid'] ==
                             unditherfa['targetid']) &
                            (ditherfa['objtype'] == 'TGT'))
                ontarget = (ontarget &
                            (unditherfa['morphtype'] == 'PSF'))
                dfiberra = (dithra-udithra)*np.cos(np.radians(udithdec))*60*60
                dfiberdec = (dithdec-udithdec)*60*60
                if not np.all(ditherfa['FIBER'] ==
                              np.arange(len(ditherfa))):
                    raise ValueError('unexpected shape of dither file')
                dfiberra[~ontarget] = np.nan
                dfiberdec[~ontarget] = np.nan
                dfiberra = dfiberra[fiber]
                dfiberdec = dfiberdec[fiber]
                mjd = fluxhead['MJD-OBS']
                dra = dfiberra
                ddec = dfiberdec
                if lookup_wcs is not None and centralwcs is not None:
                    wcs = lookup_wcs(mjd)
                    wind = usewcspair
                    if (~np.isfinite(centralwcs['cenra'][wind]) or
                        ~np.isfinite(centralwcs['cendec'][wind])):
                        raise ValueError('central ra/dec is NaN!')
                    dtelra = (wcs['cenra'][wind]-centralwcs['cenra'][wind])
                    dtelra *= np.cos(np.radians(centralwcs['cendec'][wind]))
                    dteldec = wcs['cendec'][wind]-centralwcs['cendec'][wind]
                    dra += dtelra*60*60
                    ddec += dteldec*60*60
                    if np.all(~np.isfinite(dra)):
                        print('warning: no good telescope offset, %s' %
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
                    wavebounds = dict(B=[4000, 5500],
                                      R=[5650, 7120],
                                      Z=[8500, 9950])
                    mask = (mask & (wave > wavebounds[camera][0]) &
                            (wave < wavebounds[camera][1]))
                    specflux = np.trapz(flux*mask, wave)
                    sumivar = np.sum(ivar[mask]**-1)
                    if sumivar > 0:
                        specflux_ivar = 1./np.sum(ivar[mask]**-1)
                    else:
                        specflux_ivar = 0
                    specflux /= exptime
                    specflux_ivar *= exptime**2
                tabrows.append((expid, exptime, mjd,
                                target_id[j], target_ra[j], target_dec[j],
                                fiber[j], objtype[j],
                                flux_g[j], flux_r[j], flux_z[j],
                                specflux, specflux_ivar, camera,
                                dra[j], ddec[j],
                                x[j], y[j]))

    tab = Table(rows=tabrows,
                names=('EXPID', 'EXPTIME', 'MJD_OBS',
                       'TARGETID', 'TARGET_RA', 'TARGET_DEC',
                       'FIBER', 'OBJTYPE',
                       'FLUX_G', 'FLUX_R', 'FLUX_Z',
                       'SPECTROFLUX', 'SPECTROFLUX_IVAR', 'CAMERA',
                       'DELTA_X_ARCSEC', 'DELTA_Y_ARCSEC',
                       'XFOCAL', 'YFOCAL'),
                meta={'EXTNAME' : 'DITHER',
                      'TILEID' : '{}'.format(tileid)})

    return tab


def getfilenames(expid, date, filetype, location):
    """Return a list of exposures and filenames given an INI configuration.
    Returns
    -------
    exfiles : dict
        Dictionary of exposure numbers and corresponding nightwatch
        qcframe or redux sframe FITS files.
    """

    # Set up the path and file prefix depending on the filetype.
    if filetype == 'nightwatch':
        fileprefix = 'qcframe'

        if location == 'nersc':
            prefix = '/global/project/projectdirs/desi/spectro/nightwatch/kpno'
        elif location == 'kpno':
            prefix = '/exposures/desi' # not correct path!
        else:
            raise ValueError('Unknown location {}'.format(location))
    elif filetype == 'redux':
        fileprefix = 'sframe'

        if location == 'nersc':
            prefix = '/global/project/projectdirs/desi/spectro/redux/daily/exposures'
        elif location == 'kpno':
            prefix = '/exposures/desi' # not correct path!
        else:
            raise ValueError('Unknown location {}'.format(location))
    else:
        raise ValueError('Unknown file type {}'.format(filetype))

    # Find the exposures files.
    exfiles = {}
    for ex in expid:
        folder = '{}/{}/{:08d}'.format(prefix, date, ex)
        files = sorted(glob('{}/{}*.fits'.format(folder, fileprefix)))
        exfiles[ex] = files

    return exfiles
