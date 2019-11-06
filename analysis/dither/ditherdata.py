from configparser import ConfigParser
from glob import glob

from astropy.table import Table
from astropy.io import fits

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

        # Construct fiber data.

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

    def __str__(self):
        output = []
        for ex, files in self._exposure_files.items():
            filenames = '- exposure {:08d}\n'.format(ex)
            for f in files:
                filenames = '{}  + {}\n'.format(filenames, f)
            output.append(filenames)

        return '\n'.join(output)
