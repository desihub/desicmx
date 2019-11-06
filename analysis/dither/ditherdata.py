from configparser import ConfigParser
from glob import glob

from astropy.table import Table
from astropy.io import fits

import numpy as np


class DitherSequence:
    """Access class to dithering sequence data from nightwatch or redux
    files."""

    def __init__(self, inifile):
        """Parse a configuration file in INI format.

        Parameters
        ----------
        inifile : str
            Name of INI file with configuration data.
        """

        config = ConfigParser()
        config.read(inifile)
        sequence = config['dithersequence']

        # Set up the file type and exposure sequence.
        self._location = sequence['location']
        self._filetype = sequence['filetype']
        self._date = sequence['date']
        self._exposures = sorted([int(e) for e in sequence['exposures'].split()])

        # Extract the list of exposures on disk.
        self._exposure_files = self._getfilenames()

        # Construct fiber output.
        self._exposure_table = self._buildtable()
        self._exposure_table.write('test.fits', overwrite=True)

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

        for expid, exfiles in self._exposure_files.items():
            specflux_b, specflux_r, specflux_z = [], [], []
            tab = None

            if len(exfiles) == 0:
                continue

            print(expid)
            for exfile in exfiles:
                print(exfile)
                hdu = fits.open(exfile)

                # Assume sframe format.
                wave = hdu['WAVELENGTH'].data
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

                for i, fiber_id in enumerate(fiber):
                    flux = fluxdata[i]
                    if camera == 'B':
                        specflux_b.append(np.trapz(flux, wave))
                    elif camera == 'R':
                        specflux_r.append(np.trapz(flux, wave))
                    else:
                        specflux_z.append(np.trapz(flux, wave))

            for j in range(len(fiber)):
                tabrows.append((expid,
                                target_id[j], target_ra[j], target_dec[j],
                                fiber[j], objtype[j], flux_r[j],
                                specflux_b[j], specflux_r[j], specflux_z[j],
                                x[j], y[j]))

            break

        tab = Table(rows=tabrows,
                    names=('EXPID', 'TARGETID', 'TARGET_RA', 'TARGET_DEC',
                           'FIBER', 'OBJTYPE', 'FLUX_R',
                           'SPECFLUX_B', 'SPECFLUX_R', 'SPECFLUX_Z',
                           'XFOCAL', 'YFOCAL'),
                    meta={'EXTNAME' : 'DITHER'})

        return tab

    def __str__(self):
        output = []
        for ex, files in self._exposure_files.items():
            filenames = '- exposure {:08d}\n'.format(ex)
            for f in files:
                filenames = '{}  + {}\n'.format(filenames, f)
            output.append(filenames)

        return '\n'.join(output)


