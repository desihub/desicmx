import os
import numpy as np
from astropy.io import fits

def massage(dat):
    dat = dat.copy()
    off = np.hypot(dat['delta_x_arcsec'], dat['delta_y_arcsec'])
    m = ~np.all(off < 10, axis=1)
    dat['delta_x_arcsec'][m, :] = np.nan
    dat['delta_y_arcsec'][m, :] = np.nan
    m = dat['hmag'][:, 0] < 0
    dat['hmag'][m, :] = 0
    var = 1./np.clip(dat['spectroflux_ivar'], 1e-6, np.inf)
    var += (0.05*dat['spectroflux'])**2
    dat['spectroflux_ivar'] = 1/var
    dat = fits_to_simple2d(dat)
    unusable = (np.all(~np.isfinite(dat['delta_x_arcsec']) |
                       ~np.isfinite(dat['delta_y_arcsec']), axis=1) |
                (dat['hmag'][:, 0] <= 0) |
                (dat['xfocal'][:, 0] < -800) |
                (dat['yfocal'][:, 0] < -800))
    dat = dat[~unusable, :]
    # dat['mjd_obs'] = dat['expid']
    return dat


def fits_to_simple2d(dat):
    outshape = dat['spectroflux'].shape
    def convert(dt):
        x = dt[0]
        y = dt[1]
        if x == 'mjd_obs':
            y = 'f8'
        return (x, y)

    outdtype = [convert(x[:2]) for x in dat.dtype.descr]

    out = np.zeros(outshape, dtype=outdtype)
    for field in dat.dtype.names:
        out[field] = dat[field]
    return out


# Conor 5.3
fieldtofpc_pparam = np.array(
    [('APO', 1.66, 8939, 8163, 1.40708, 6.13779e-03, 7.25138e-04, -3.28007e-06, -1.65995e-05),
     ('APO', 0.54, 9208, 8432, 1.36580, 6.09425e-03, 6.54926e-04, 2.62176e-05, -2.27106e-05),
     ('APO', 0.6231, 9164, 8388, 1.37239, 6.09825e-03, 6.67511e-04, 2.14437e-05, -2.17330e-05),
     ('LCO', 1.66, 8905, 7912, 2.11890, 1.40826e-02, 1.27996e-04, 6.99967e-05, 0),
     ('LCO', 0.54, 9938, 8945, 1.89824, 1.31773e-02, 1.04445e-04, 5.77341e-05, 0),
     ('LCO', 0.6231, 9743, 8751, 1.93618, 1.33536e-02, 9.17031e-05, 6.58945e-05, 0)],
    dtype=[('site', 'U20'), ('lambda', 'f4'), ('R', 'f4'), ('b', 'f4'),
           ('c0', 'f4'), ('c1', 'f4'), ('c2', 'f4'), ('c3', 'f4'),
           ('c4', 'f4')])

# Conor 5.4
fpctofield_param = np.array(
    [('APO', 1.66, 8939, 8163, 7.10691e-01, -1.56306e-03, -8.60362e-05, 3.10036e-06, 3.16259e-07),
     ('APO', 0.54, 9208, 8432, 7.32171e-01, -1.74740e-03, -9.28511e-05, 1.80969e-06, 6.48944e-07),
     ('APO', 0.6231, 9164, 8388, 7.28655e-01, -1.71534e-03, -9.19802e-05, 2.07648e-06, 5.84442e-07),
     ('LCO', 1.66, 8905, 7912, 4.71943e-01, -6.98482e-04, 1.58969e-06, -1.47239e-07, 0),
     ('LCO', 0.54, 9938, 8945, 5.26803e-01, -1.01471e-03, 3.47109e-06, -2.98113e-07, 0),
     ('LCO', 0.6231, 9743, 8751, 5.16480e-01, -9.50007e-04, 3.34034e-06, -2.93032e-07, 0)],
    dtype=[('site', 'U20'), ('lambda', 'f4'), ('R', 'f4'), ('b', 'f4'),
           ('c0', 'f4'), ('c1', 'f4'), ('c2', 'f4'), ('c3', 'f4'),
           ('c4', 'f4')])


def phifc(radius, camera, site):
    camsitedict = dict(r=0.6231, b=0.54, h=1.66)
    m = ((fpctofield_param['site'] == site) &
         (fpctofield_param['lambda'] == camsitedict[camera]))
    if np.sum(m) != 1:
        raise ValueError('failed to find good matching focal plane parameters; '
                         'bad camera or site?')
    param = fpctofield_param[m][0]
    # I need the radial and azimuthal plate scale at a given focal plane radius
    # this is microns / arcsec.
    radcurve = param['R']
    bb = param['b']
    zfpc = -radcurve*np.cos(np.arcsin(radius/radcurve))+bb
    X = (bb-zfpc)/np.sqrt(radius**2 + (zfpc - bb)**2)
    # doing this analytically is unpleasant and likely to be wrong.
    phifpc = np.degrees(np.arccos(X))
    phifc = (param['c0']*phifpc + param['c1']*phifpc**3 +
             param['c2']*phifpc**5 + param['c3']*phifpc**7 +
             param['c4']*phifpc**9)
    return phifc


def platescale(camera, site):
    out = np.zeros(301, dtype=[
        ('radius', 'f4'),
        ('radial_platescale', 'f4'),
        ('az_platescale', 'f4')])
    radius = np.linspace(0, 300, 301)
    phi_sky = phifc(radius, camera, site)
    ps = np.gradient(radius, phi_sky)
    radps = 1000*ps/3600
    azps = radps*0
    m = radius > 0
    azps[m] = radius[m] / np.sin(np.radians(phi_sky[m]))
    azps = 1000*azps*(np.pi/180)/3600
    azps[~m] = azps[m][0]
    out['radius'] = radius
    out['radial_platescale'] = radps
    out['az_platescale'] = azps
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze SDSS dither data.')
    parser.add_argument('filename', help='dither summary file to analyze')
    parser.add_argument('-l', '--label', default=None, type=str,
                        help='label for file names')
    parser.add_argument('-n', '--nthreads', default=0, type=int,
                        help='number of threads to use')
    args = parser.parse_args()
    if args.label is None:
        label = args.filename.split('.')[0]+'%s'
    else:
        label = args.label
    dat = fits.getdata(args.filename)
    dat2 = massage(dat)
    if len(np.unique(dat['camera'])) != 1:
        raise ValueError('multiple camera values in file')
    cam = dat['camera'][0, 0]
    cam = 'h' if cam == 'APOGEE' else cam
    platescalesdss = platescale(cam[0], 'APO')
    label = args.label
    if label is None:
        outname = os.path.basename(args.filename)[:-5]
        label = outname+'-analysis-%s'
    import solvedither
    solvedither.process({cam: dat2}, cam,
                        overwrite=True, label=label,
                        threads=args.nthreads, platescale=platescalesdss,
                        fiberdiameter=120, useguess=False)
