# Dither

This area contains basic modules to collect data from dither sequences, either
from the output of Nightwatch or the full spectroscopic pipeline (redux).

Proposed output format (Eddie S. and David S.):

    EXPID >i4
    TARGET_RA ('>f8', (5000,))
    TARGET_DEC ('>f8', (5000,))
    FIBER ('>i4', (5000,))
    OBJTYPE ('S3', (5000,))
    FLUX_R ('>f4', (5000,))
    SPECTROFLUX ('>f4', (5000,))
    DELTA_X_ARCSEC ('>f4', (5000,))
    DELTA_Y_ARCSEC ('>f4', (5000,))
    XFOCAL ('>f4', (5000,))
    YFOCAL ('>f4', (5000,))
    TRUTH_FWHM >f4
    TRUTH_TRANSPARENCY >f4
    TRUTH_XTELOFFSET >f4
    TRUTH_YTELOFFSET >f4
    TRUTH_XFIBOFFSET ('>f4', (5000,))
    TRUTH_YFIBOFFSET ('>f4', (5000,))

Additional output fields may include:

    TARGETID
    SPECTROFLUX_IVAR
