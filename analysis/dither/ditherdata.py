from configparser import ConfigParser
from glob import glob

from astropy.table import Table
from astropy.io import fits

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
        self._exposures = sorted([int(e) for e in sequence['exposures'].split()])

        coords = config['coordinates']
        self._deltara = [float(d) for d in coords['deltara'].split()]
        self._deltadec = [float(d) for d in coords['deltadec'].split()]

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

            dra  = self._deltara[i]
            ddec = self._deltadec[i]

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

                target_id = fibermap['TARGETID']
                target_ra = fibermap['TARGET_RA']
                target_dec = fibermap['TARGET_DEC']
                fiber = fibermap['FIBER']
                objtype = fibermap['OBJTYPE']
                flux_r = fibermap['FLUX_R']
                x, y = [fibermap['FIBERASSIGN_{}'.format(val)] for val in ('X', 'Y')]

                camera = fluxhead['CAMERA'][0].upper()

                for j, fiber_id in enumerate(fiber):
                    flux = fluxdata[j]
                    if camera == 'B':
                        specflux_b.append(np.trapz(flux, wave))
                    elif camera == 'R':
                        specflux_r.append(np.trapz(flux, wave))
                    else:
                        specflux_z.append(np.trapz(flux, wave))

            for k in range(len(fiber)):
                tabrows.append((expid,
                                target_id[k], target_ra[k], target_dec[k],
                                fiber[k], objtype[k], flux_r[k],
                                specflux_b[k], specflux_r[k], specflux_z[k],
                                dra, ddec,
                                x[k], y[k]))

        tab = Table(rows=tabrows,
                    names=('EXPID', 'TARGETID', 'TARGET_RA', 'TARGET_DEC',
                           'FIBER', 'OBJTYPE', 'FLUX_R',
                           'SPECFLUX_B', 'SPECFLUX_R', 'SPECFLUX_Z',
                           'DELTA_X_ARCSEC', 'DELTA_Y_ARCSEC',
                           'XFOCAL', 'YFOCAL'),
                    meta={'EXTNAME' : 'DITHER'})

        return tab

    def save(self, filename=None, overwrite=True):
        """Save exposure table to a FITS file.

        Parameters
        ----------
        filename : str
            Output filename.
        overwrite : bool
            If true, clobber an existing file with the same name.
        """
        if filename is None:
            filename = self._output
        self._exposure_table.write(filename, overwrite=overwrite)

    def __str__(self):
        """String representation of the exposure sequence.
        """
        output = []
        for ex, files in self._exposure_files.items():
            filenames = '- exposure {:08d}\n'.format(ex)
            for f in files:
                filenames = '{}  + {}\n'.format(filenames, f)
            output.append(filenames)

        return '\n'.join(output)


