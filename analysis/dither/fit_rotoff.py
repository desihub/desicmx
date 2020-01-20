import os
import numpy as np
from astropy.io import fits
from astropy.io import ascii
import desimeter
import argparse

def lb2tan(l, b, lcen=None, bcen=None):
    up = np.array([0, 0, 1])
    uv = lb2uv(l, b)
    if lcen is None:
        lcen, bcen = uv2lb(np.mean(uv, axis=0).reshape(1, -1))
        lcen, bcen = lcen[0], bcen[0]
    uvcen = lb2uv(lcen, bcen)
    # error if directly at pole
    rahat = np.cross(up, uvcen)
    rahat /= np.sqrt(np.sum(rahat**2))
    dechat = np.cross(uvcen, rahat)
    dechat /= np.sqrt(np.sum(dechat**2))
    xx = np.einsum('i,ji', rahat, uv)
    yy = np.einsum('i,ji', dechat, uv)
    xx *= 180/np.pi
    yy *= 180/np.pi
    return xx, yy


def lb2tp(l, b):
    return (90.-b)*np.pi/180., l*np.pi/180.


def tp2lb(t, p):
    return p*180./np.pi % 360., 90.-t*180./np.pi


def lb2uv(r, d):
    return tp2uv(*lb2tp(r, d))


def uv2lb(uv):
    return tp2lb(*uv2tp(uv))


def uv2tp(uv):
    norm = np.sqrt(np.sum(uv**2., axis=1))
    uv = uv / norm.reshape(-1, 1)
    t = np.arccos(uv[:,2])
    p = np.arctan2(uv[:,1], uv[:,0])
    return t, p


def tp2uv(t, p):
    z = np.cos(t)
    x = np.cos(p)*np.sin(t)
    y = np.sin(p)*np.sin(t)
    return np.concatenate([q[...,np.newaxis] for q in (x, y, z)],
                             axis=-1)


def mjd2lst(mjd, lng):
    """ Stolen from ct2lst.pro in IDL astrolib.
    Returns the local sidereal time at a given MJD and longitude. """

    mjdstart = 2400000.5
    jd = mjd + mjdstart
    c = [280.46061837, 360.98564736629, 0.000387933, 38710000.0 ]
    jd2000 = 2451545.0
    t0 = jd - jd2000
    t = t0/36525.
    theta = c[0] + (c[1] * t0) + t**2*(c[2] - t/ c[3] )
    lst = (theta + lng)/15.
    lst = lst % 24.
    return lst


def match(a, b):
    sa = np.argsort(a)
    sb = np.argsort(b)
    ua = np.unique(a)
    ub = np.unique(b)
    if len(ua) != len(a):# or len(ub) != len(b):
        raise ValueError('All keys in a must be unique.')
    ind = np.searchsorted(a[sa], b)
    m = (ind >= 0) & (ind < len(a))
    matches = a[sa[ind[m]]] == b[m]
    m[m] &= matches
    return sa[ind[m]], np.flatnonzero(m)


def transform(param, pts):
    newpts = pts.copy()
    newpts[:, 0] = pts[:, 0] + param[0]
    newpts[:, 1] = pts[:, 1] + param[1]
    newpts *= param[2]
    theta = np.radians(param[3])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    newpts = rot.dot(newpts.T).T
    return newpts


def read_idealfp():
    res = np.zeros(10, dtype=[('X', 'f8'), ('Y', 'f8'), ('GFA_LOC', 'i4')])
    res['GFA_LOC'] = np.arange(10)
    ang = np.arange(10)*2*np.pi/10
    scale = 407.34  # average from metrology file for pinhole_id == 1
    res['X'] = np.sin(ang)*scale
    res['Y'] = -np.cos(ang)*scale
    return res


def read_gfa(fn=None, detailed=True):
    guiders = ['GUIDE%d' % i for i in [0, 2, 3, 5, 7, 8]]
    from astropy.wcs import WCS
    out = np.zeros(len(guiders), dtype=[('RA', 'f8'), ('DEC', 'f8'),
                                           ('X', 'f8'), ('Y', 'f8'),
                                           ('GFA_LOC', 'i4')])
    if fn is None:
        average = True
        fn = os.path.join(os.environ['GFA_REDUCE_ETC'],
                          'dummy_with_headers_gaia.bigtan.fits.gz')
    else:
        average = False

    for i, guider in enumerate(guiders):
        h = fits.getheader(fn, guider)
        wcs = WCS(h)
        out['GFA_LOC'][i] = int(guider[-1])
        r, d = wcs.all_pix2world(1, 1, 1)
        out['RA'][i] = r
        out['DEC'][i] = d
    if average or not detailed:
        xx, yy = lb2tan(out['RA'], out['DEC'])
        xx = -xx*270
        yy = yy*270
        out['X'] = xx
        out['Y'] = yy
    else:
        uv = lb2uv(out['RA'], out['DEC'])
        uvcen = np.mean(uv, axis=0)
        lcen, bcen = uv2lb(uvcen.reshape(1, -1))
        lcen, bcen = lcen[0], bcen[0]
        mjd = fits.getdata(fn.replace('_reduced', '_ccds'))['mjd'][0]
        lst = mjd2lst(mjd, -111.599208)*360/24.
        from desimeter.transform.radec2tan import radec2tan
        xx, yy = radec2tan(out['RA'], out['DEC'], lcen, bcen, mjd, lst, 0,
                           precession=True, aberration=True)
        xx *= 270*180/np.pi
        yy *= 270*180/np.pi
        out['X'] = xx
        out['Y'] = yy
    return out


def read_fp():
    if 'FP_METROLOGY' in os.environ:
        fn = os.environ['FP_METROLOGY']
    else:
        fn = os.path.join(
            os.environ['HOME'],
            'desi/desihub/desimeter/py/desimeter/data/fp-metrology.csv')
    res = ascii.read(fn)
    return res


def fit_points(x1, y1, x2, y2):
    c1 = np.array([x1, y1]).T
    c2 = np.array([x2, y2]).T
    def chi(par):
        return (c1-transform(par, c2)).reshape(-1)
    from scipy.optimize import leastsq
    res = leastsq(chi, [0, 0, 1, 0], full_output=True, epsfcn=1e-5)
    return res


def fit_gfa(fn, detailed=True):
    gfa = read_gfa(fn, detailed=detailed)
    fp = read_fp()
    mfp = (fp['DEVICE_TYPE'] == 'GFA') & (fp['PINHOLE_ID'] == 1)
    fp = fp[mfp]
    gfa_to_use = []
    for i in range(10):
        if (i in gfa['GFA_LOC']) and (i in fp['PETAL_LOC']):
            gfa_to_use.append(i)
    fp = fp[np.array([loc in gfa_to_use for loc in fp['PETAL_LOC']])]
    gfa = gfa[np.array([loc in gfa_to_use for loc in gfa['GFA_LOC']])]
    fp = fp[np.argsort(fp['PETAL_LOC'])]
    return fit_points(gfa['X'], gfa['Y'], fp['X_FP'], fp['Y_FP'])


def do_fits(gfa_to_use=None):
    gfa = read_average_gfa()
    fp = read_fp()
    ideal = read_idealfp()
    mfp = (fp['DEVICE_TYPE'] == 'GFA') & (fp['PINHOLE_ID'] == 1)
    fp = fp[mfp]
    if gfa_to_use is None:
        gfa_to_use = []
        for i in range(10):
            if (i in gfa['GFA_LOC']) and (i in fp['PETAL_LOC']):
                gfa_to_use.append(i)
    print('Using GFAs '+' '.join(str(g) for g in gfa_to_use))
    gfa = gfa[np.array([loc in gfa_to_use for loc in gfa['GFA_LOC']])]
    fp = fp[np.array([loc in gfa_to_use for loc in fp['PETAL_LOC']])]
    ideal = ideal[np.array([loc in gfa_to_use for loc in ideal['GFA_LOC']])]
    gfa = gfa[np.argsort(gfa['GFA_LOC'])]
    fp = fp[np.argsort(fp['PETAL_LOC'])]
    ideal = ideal[np.argsort(ideal['GFA_LOC'])]
    # should be three matched sets of identical GFAs in different frames.
    xi = np.array([ideal['X'], ideal['Y']]).T
    xf = np.array([fp['X_FP'], fp['Y_FP']]).T
    xg = np.array([gfa['X'], gfa['Y']]).T

    def chi(param, x1, x2):
        return (x1-transform(param, x2)).reshape(-1)

    from scipy.optimize import leastsq
    guess = [0, 0, 1, 0]
    results = []
    for x1, x2 in [(xi, xf), (xi, xg), (xf, xg)]:
        res = leastsq(chi, guess, args=(x1, x2), full_output=True,
                      epsfcn=1e-5)
        model = transform(res[0], x2)
        data = x1
        print(x1)
        results.append([res[0], data, model])
        print(np.sqrt(np.sum((model-data)**2)/3))
    return results


# ra dec -> tangent plane
# fit FP <-> radec
# predict radec for GFA
# fit GFA predicted radec to measured radec


def fit_rotoff(fvcfn, coordfn, fafn, gfafn, verbose=True):
    fa, fahdr = fits.getdata(fafn, 'FIBERASSIGN', header=True)
    gfa = read_gfa(gfafn)
    fvc = ascii.read(fvcfn)
    coord = fits.getdata(coordfn)
    fp = read_fp()
    lcen, bcen = fahdr['TILERA'], fahdr['TILEDEC']
    m = coord['FLAGS_FVC_0'] == 4
    # should be things like FLAGS_FVC_3, but none in example file
    # I'm looking at.
    coord = coord[m]
    coordloc = 1000*coord['PETAL_LOC']+coord['DEVICE_LOC']
    mc, mf = match(coordloc, fvc['LOCATION'])
    fvc = fvc[mf]
    # fibers with FLAGS_FVC_0 != 4 are gone.
    mfa, mfvc = match(fa['LOCATION'], fvc['LOCATION'])
    xx, yy = lb2tan(fa['TARGET_RA'], fa['TARGET_DEC'],
                    lcen=lcen, bcen=bcen)
    from desimeter.transform.fvc2fp.poly2d import _polyfit2d, _vander2d
    from astropy.stats import mad_std
    keep = np.ones(len(mfvc), dtype='bool')
    degree = 5
    for i in range(3):
        px, py = _polyfit2d(fvc['X_FP'][mfvc[keep]], fvc['Y_FP'][mfvc[keep]],
                            xx[mfa[keep]], yy[mfa[keep]], degree=degree)
        A = _vander2d(fvc['X_FP'][mfvc], fvc['Y_FP'][mfvc],
                      degree).T
        fitx = A.dot(px)
        fity = A.dot(py)
        xmed = np.median((fitx-xx[mfa])[keep])
        ymed = np.median((fity-yy[mfa])[keep])
        xstd = mad_std((fitx-xx[mfa])[keep])
        ystd = mad_std((fity-yy[mfa])[keep])
        keep = (keep & (np.abs(fitx - xx[mfa] - xmed) < 10*xstd) &
                (np.abs(fity - yy[mfa] - ymed) < 10*ystd))
    if verbose:
        print('FVC -> radec accuracy: %5.2f %5.2f arcsec' %
              (xstd*60*60, ystd*60*60))
    # now we have FP -> tan (-> radec)
    # now we want GFA positions
    m = (fp['DEVICE_TYPE'] == 'GFA') & (fp['PINHOLE_ID'] == 1)
    fpgfa = fp[m]
    usegfas = np.flatnonzero(np.array(
        [loc in fpgfa['PETAL_LOC'] and loc in gfa['GFA_LOC']
         for loc in range(10)]))
    fpgfa = fpgfa[np.array([x in usegfas for x in fpgfa['PETAL_LOC']])]
    fpgfa = fpgfa[np.argsort(fpgfa['PETAL_LOC'])]
    gfa = gfa[np.array([x in usegfas for x in gfa['GFA_LOC']])]
    gfa = gfa[np.argsort(gfa['GFA_LOC'])]
    A = _vander2d(fpgfa['X_FP'], fpgfa['Y_FP'], degree).T
    xgfatanfa = A.dot(px)
    ygfatanfa = A.dot(py)
    # these are the predicted locations of the GFAs in the tangent
    # plane
    xgfatanmeas, ygfatanmeas = lb2tan(gfa['RA'], gfa['DEC'],
                                      lcen=lcen, bcen=bcen)
    res = fit_points(xgfatanfa, ygfatanfa, xgfatanmeas, ygfatanmeas)
    model = transform(res[0], np.array([xgfatanmeas, ygfatanmeas]).T)
    xmod, ymod = model[:, 0], model[:, 1]
    xstd = np.std(xmod - xgfatanfa)*60*60
    ystd = np.std(ymod - ygfatanfa)*60*60
    if verbose:
        print('FA GFA pos -> measured GFA pos, RMS, arcsec, '
              'after fit: %5.2f %5.2f' % (xstd, ystd))
    xtel, ytel, scale, rot = res[0]
    xtel = xtel*60*60
    ytel = ytel*60*60
    if verbose:
        print(f'Telescope needs correction by:'
              f'\nx:{xtel:+6.2f}"\ny:{ytel:+6.2f}"\n'
              f'(scale:{scale:+9.5f})\nrotation:{rot:+7.3f} deg')
    return ((xgfatanfa, ygfatanfa), (xgfatanmeas, ygfatanmeas),
            (xmod, ymod), res)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find rotation & offset from observation')
    parser.add_argument('-f', '--fafn', type=str, default=None,
                        required=True,
                        help='path to fiberassign file observed')
    parser.add_argument('-g', '--gfafn', type=str, default=None,
                        required=True,
                        help='path to gfa_reduce reduced image file')
    parser.add_argument('-c', '--coordinatesfn', type=str, default=None,
                        required=True,
                        help='path to coordinates file')
    parser.add_argument('-d', '--desimeterfvcfn', type=str, default=None,
                        required=True,
                        help='path to desimeter fvc analysis')
    args = parser.parse_args()

    fit_rotoff(args.desimeterfvcfn, args.coordinatesfn, args.fafn, args.gfafn,
               verbose=True)
