import os
import glob
import numpy as np
import pdb
from scipy.optimize import leastsq
import desimodel.io
from astropy.io import fits

# can we try to eliminate some of the arguments?
# the data: flux, dflux, delta_x_arcsec, delta_y_arcsec, xfocal, yfocal
# data = np.zeros((n_fib, n_exp), 
#   dtype=[('spectroflux', 'f4'), ('spectroflux_ivar', 'f4'), 
#          ('delta_x_arcsec', 'f4'), ('delta_y_arcsec', 'f4'),
#          ('xfocal', 'f4'), ('yfocal', 'f4')])

# currently ~5x slower than necessary because we loop through and make
# ten separate calls to gaussian() rather than a vectorized call
# because we work through the psf object.
# a bit tricky data model to make this vectorized.  Maybe we don't care.

def gaussian(x, y, fwhm):
    sigma = fwhm/np.sqrt(8*np.log(2))
    psf =  ((2*np.pi*sigma**2)**(-1)*
            np.exp(-0.5*(x**2/sigma**2+y**2/sigma**2)))
    return psf


def moffat(x, y, fwhm, beta=3.5):
    alpha = fwhm/(2*np.sqrt(2**(1./beta)-1))
    rr2 = x**2+y**2
    return (beta - 1)/(np.pi*alpha**2)*(1+rr2/alpha**2)**(-beta)


class SimplePSF(object):
    def __init__(self, fwhm, psffun=gaussian):
        self.fwhm = fwhm
        self.psffun = psffun
    
    def fiberfrac(self, x, y, dx, dy):
        return self.psffun(dx, dy, self.fwhm)


# let's say measurements are an n_fib x n_exp array
def model_flux(data, starflux, dx, dy, psf, transparency):
    if not isinstance(psf, list):
        psf = [psf]
    fiberfracs = np.array(
        [psf0.fiberfrac(data['xfocal'][:, i], data['yfocal'][:, i], 
                        dx[:, i], dy[:, i])
         for (i, psf0) in enumerate(psf)]).T
    fluxes = fiberfracs*transparency.reshape(1, -1)*starflux.reshape(-1, 1)
    return fluxes


def model_flux_full(data, starflux, xfiboff, yfiboff, xtel, ytel, psf, 
                    transparency):
    dx = xfiboff.reshape(-1, 1) + data['delta_x_arcsec'] + xtel.reshape(1, -1)
    dy = yfiboff.reshape(-1, 1) + data['delta_y_arcsec'] + ytel.reshape(1, -1)
    flux = model_flux(data, starflux, dx, dy, psf, transparency)
    return flux


def chi2andgrad(data, starflux, xfiboff, yfiboff, xtel, ytel, fwhm, transparency,
                psffun, hfwhm=1e-3, hoffset=1e-3, damp=5):
    if len(np.atleast_1d(fwhm)) > 1:
        psf = [psffun(fwhm0) for fwhm0 in fwhm]
        psfh = [psffun(fwhm0 + hfwhm) for fwhm0 in fwhm]
    else:
        psf = [psffun(fwhm)]
        psfh = [psffun(fwhm + hfwhm)]
    modflux = model_flux_full(data, starflux, xfiboff, yfiboff, xtel, ytel,
                              psf, transparency)
    chi = (data['spectroflux']-modflux)*np.sqrt(data['spectroflux_ivar'])
    # nparam = 3*nfiber+4*nexp
    # grad = np.zeros(nparam, dtype='f4')
    # need del model / dx.
    dmdflux = (modflux/(starflux[:, None] + (starflux[:, None] == 0))*
               np.sqrt(data['spectroflux_ivar']))
    dmdtrans = (modflux/(transparency[None, :] + (transparency[None, :] == 0))*
               np.sqrt(data['spectroflux_ivar']))
    dmdxfiboff = (model_flux_full(data, starflux, xfiboff+hoffset, yfiboff, 
                                xtel, ytel, psf, transparency) -
                modflux)*np.sqrt(data['spectroflux_ivar'])/hoffset
    dmdyfiboff = (model_flux_full(data, starflux, xfiboff, yfiboff+hoffset, 
                                xtel, ytel, psf, transparency) -
                modflux)*np.sqrt(data['spectroflux_ivar'])/hoffset
    dmdfwhm = (model_flux_full(data, starflux, xfiboff, yfiboff,
                               xtel, ytel, psfh, transparency) -
               modflux)*np.sqrt(data['spectroflux_ivar'])/hfwhm
    grad = {}
    if damp == 0:
        dfdchi = 1
    else:
        dfdchi = damper_deriv(chi, damp)
        chi = damper(chi, damp)
    chi2 = np.sum(chi**2)
    grad['starflux'] = -2*np.sum(chi*dmdflux*dfdchi, axis=1)
    grad['xfiboff'] = -2*np.sum(chi*dmdxfiboff*dfdchi, axis=1)
    grad['xtel'] = -2*np.sum(chi*dmdxfiboff*dfdchi, axis=0)
    grad['yfiboff'] = -2*np.sum(chi*dmdyfiboff*dfdchi, axis=1)
    grad['ytel'] = -2*np.sum(chi*dmdyfiboff*dfdchi, axis=0)
    grad['fwhm'] = -2*np.sum(chi*dmdfwhm*dfdchi, axis=0)
    grad['transparency'] = -2*np.sum(chi*dmdtrans*dfdchi, axis=0)
    return chi2, grad


def fit_all(data, guessflux, psffun=SimplePSF,
            truth=None, verbose=False, **kw):

    nfib, nexp = data.shape
    paramdat = [('starflux', nfib),
                ('xfiboff', nfib),
                ('yfiboff', nfib),
                ('xtel', nexp),
                ('ytel', nexp),
                ('fwhm', nexp),
                ('transparency', nexp)]

    fivesig = data['spectroflux']*np.sqrt(data['spectroflux_ivar']) > 5
    fluxscale = np.median(data['spectroflux'][fivesig])
    datascale = data.copy()
    datascale['spectroflux'] /= fluxscale
    datascale['spectroflux_ivar'] *= fluxscale**2

    def chi2andgradwrap(param, returngrad=True, prior=None):
        nfib, nexp = datascale.shape
        posargs = [datascale]
        nsofar = 0
        for paramname, paramnum in paramdat:
            posargs.append(param[nsofar:nsofar+paramnum])
            nsofar += paramnum
        posargs.append(psffun)
        chi2, grad = chi2andgrad(*posargs, **kw)
        grad = np.concatenate([grad[name] for (name, _) in paramdat])
        if prior is not None:
            priora, priorb, priorsig = prior
            chi = (priorb-priora.dot(param))/priorsig
            chi2 += np.sum(chi**2)
            grad += -2*priora.T.dot(chi/priorsig)
        # print(chi2)
        if returngrad:
            return chi2, grad
        else:
            return chi2

    guess = initial_guess(datascale, guessflux/fluxscale, psffun)
    guessfib = [guessflux/fluxscale, guess['xfiboff'], guess['yfiboff']]
    guessim = [np.ones(nexp, dtype='f4')*guess['xtel'], 
               np.ones(nexp, dtype='f4')*guess['ytel'],
               np.ones(nexp, dtype='f4')*guess['fwhm'],
               np.ones(nexp, dtype='f4')*guess['transparency']]
    guess = np.concatenate(guessfib+guessim)
    pdb.set_trace()

    from scipy.optimize import minimize
    res = minimize(chi2andgradwrap, guess, method='CG', jac=True)
    return res


def chi_image(param, data=None, dflux=None, starflux=None, 
              xfiboff=None, yfiboff=None, psffun=SimplePSF,
              **extra):
    xtel = param[0:1]
    ytel = param[1:2]
    fwhm = param[2:3]
    transparency = param[3:4]
    psf = psffun(fwhm)
    modflux = model_flux_full(data, starflux, xfiboff, yfiboff, 
                              xtel, ytel, psf, transparency)
    # add some really weak priors
    weakpriors = np.array([xtel/3600, ytel/3600, 
                              (fwhm-1)/100, (transparency-0.4)/10])
    chi = (data['spectroflux']-modflux)*np.sqrt(data['spectroflux_ivar'])
    return np.concatenate([chi.reshape(-1), weakpriors.reshape(-1)])


def chi_fiber(param, data=None,
              xtel=None, ytel=None, psf=None, transparency=None, **extra):
    starflux = param[0:1]
    xfiboff = param[1:2]
    yfiboff = param[2:3]
    modflux = model_flux_full(data, starflux, xfiboff, yfiboff,
                              xtel, ytel, psf, transparency)
    # add some really weak priors
    weakpriors = np.array([xfiboff/100, yfiboff/100, starflux/10**6])
    chi = (data['spectroflux']-modflux)*np.sqrt(data['spectroflux_ivar'])
    return np.concatenate([chi.reshape(-1), weakpriors.reshape(-1)])


# find the set of image parameters (xtel, ytel, fwhm, transparency)
# that lead model_flux_full to most closely approximate flux
# flux, dflux are n_fib x n_exp
def fit_images(data, starflux, 
               xfiboff, yfiboff, psffun=None, guess=None, truth=None):

    def chi(param, args=None):
        return damper(chi_image(param, **args), 5)

    nfiber, nexp = data.shape
    nparam = 4  # xtel, ytel, fwhm, transparency

    if guess is None:
        guess = np.array([0, 0, 1.0, 0.4], dtype='f4')
        guess = guess.reshape(1, -1)*np.ones((nexp, 1), dtype='f4')

    outpar = np.zeros((nexp, nparam), dtype='f4')
    outunc = np.zeros((nexp, nparam), dtype='f4')
    bestchi = data['spectroflux']*0
    for i in range(nexp):
        args = dict(data=data[:, i:i+1],
                    starflux=starflux, 
                    xfiboff=xfiboff, yfiboff=yfiboff,
                    psffun=psffun)
        res = leastsq(chi, guess[i], args=(args,), full_output=True)
        outpar[i, :] = res[0]
        if res[1] is not None:
            outunc[i, :] = np.diag(res[1])
        else:
            outunc[i, :] = np.inf
        bestchi[:, i] = chi(res[0], args=args)[:nfiber]
    return outpar, outunc, bestchi


# find the set of fiber parameters (starflux, xfiboff, yfiboff)
# that lead model_flux_full to most closely approximate flux
# flux, dflux are n_fib x n_exp
def fit_fibers(data, xtel, ytel, fwhm, transparency, 
               psffun=None, guess=None):

    def chi(param, args=None):
        return damper(chi_fiber(param, **args), 5)

    nfiber, nexp = data.shape
    nparam = 3  # starflux, xfiboff, yfiboff

    if guess is None:
        guess = np.zeros((nfiber, nparam), dtype='f4')

    outpar = np.zeros((nfiber, nparam), dtype='f4')
    outunc = np.zeros((nfiber, nparam), dtype='f4')
    bestchi = data['spectroflux']*0
    psf = [psffun(fwhm0) for fwhm0 in fwhm]
    for i in range(nfiber):
        args = dict(data=data[i:i+1, :], 
                    xtel=xtel, ytel=ytel,
                    psf=psf, transparency=transparency)
        res = leastsq(chi, guess[i], args=(args,), full_output=True)
        outpar[i, :] = res[0]
        if res[1] is not None:
            outunc[i, :] = np.sqrt(np.diag(res[1]))
        else:
            outunc[i, :] = np.inf
        bestchi[i, :] = chi(res[0], args=args)[:nexp]
        if np.any(~np.isfinite(res[0])):
            pdb.set_trace()
    # pdb.set_trace()
    return outpar, outunc, bestchi



def fake_data(nexp=10, nfiber=5000, mseeing=1.1, devseeing=0.2, meantrans=0.4,
              devtrans=0.01,
              roff=0.2, teloff=0.1, frange=[5000, 10000], pattern='gaussian',
              ditherscale=0.25, psffun=SimplePSF,
              seed=None, sky=150):
    if seed is not None:
        np.random.seed(seed)
    if pattern == 'gaussian':
        delta_x_arcsec = np.random.randn(nfiber, nexp)*ditherscale
        delta_y_arcsec = np.random.randn(nfiber, nexp)*ditherscale
    elif pattern == 'box':
        delta_x_arcsec = (1-2*np.random.rand(nfiber, nexp))*ditherscale
        delta_y_arcsec = (1-2*np.random.rand(nfiber, nexp))*ditherscale
    elif pattern == 'rtheta':
        rcomfib = np.sqrt(np.random.rand(nfiber, nexp))*ditherscale
        thetacomfib = np.random.rand(nfiber, nexp)*2*np.pi
        delta_x_arcsec = rcomfib*np.cos(thetacomfib)
        delta_y_arcsec = rcomfib*np.sin(thetacomfib)
    elif pattern == 'cross':
        xory = np.random.rand(nfiber, nexp) > 0.5
        delta_x_arcsec = xory * (1-2*np.random.rand(nfiber, nexp))*ditherscale
        delta_y_arcsec = ~xory * (1-2*np.random.rand(nfiber, nexp))*ditherscale
    elif pattern == 'telescope':
        delta_x_tel = np.random.randn(nexp)*ditherscale
        delta_y_tel = np.random.randn(nexp)*ditherscale
        delta_x_arcsec_const = np.random.randn(nfiber)*ditherscale
        delta_y_arcsec_const = np.random.randn(nfiber)*ditherscale
        delta_x_arcsec = delta_x_arcsec_const[:, None]+delta_x_tel[None, :]
        delta_y_arcsec = delta_y_arcsec_const[:, None]+delta_y_tel[None, :]
    else:
        raise ValueError('not implemented')
    if seed is not None:
        np.random.seed(seed+1)
    import desimodel.io
    fpos = desimodel.io.load_fiberpos()
    xfocal = fpos['X']
    yfocal = fpos['Y']
    if nfiber > len(xfocal):
        raise ValueError('only 5000 fibers!')
    elif nfiber < len(xfocal):
        import random
        ind = random.sample(range(len(xfocal)), nfiber)
        xfocal = xfocal[ind]
        yfocal = yfocal[ind]
    xtel = np.random.randn(nexp)*teloff
    ytel = np.random.randn(nexp)*teloff
    fwhm = np.clip(np.random.randn(nexp)*0.2+1.1, 0.7, np.inf)
    transparency = np.random.randn(nexp)*devtrans+meantrans
    xfiboff = (np.random.rand(nfiber)-0.5)*roff
    yfiboff = (np.random.rand(nfiber)-0.5)*roff
    starflux = np.random.rand(nfiber)*(frange[1]-frange[0])+frange[0]
    starflux /= meantrans
    data = np.zeros((nfiber, nexp), dtype=[
        ('spectroflux', 'f4'), ('spectroflux_ivar', 'f4'),
        ('xfocal', 'f4'), ('yfocal', 'f4'),
        ('delta_x_arcsec', 'f4'), ('delta_y_arcsec', 'f4'),
        ('trueflux', 'f4')])
    data['xfocal'] = xfocal[:, None]
    data['yfocal'] = yfocal[:, None]
    data['delta_x_arcsec'] = delta_x_arcsec
    data['delta_y_arcsec'] = delta_y_arcsec
    psf = [psffun(fwhm0) for fwhm0 in fwhm]
    flux = model_flux_full(data, starflux, xfiboff, yfiboff,
                           xtel, ytel, psf, transparency)
    if np.any(flux == 0):
        pdb.set_trace()
    dflux = np.sqrt(flux+sky)  # additional noise from sky photons
    data['trueflux'] = flux
    data['spectroflux'] = flux + np.random.randn(*flux.shape)*dflux
    data['spectroflux_ivar'] = 1./dflux**2
    return dict(data=data,
                xtel=xtel, ytel=ytel, fwhm=fwhm, transparency=transparency,
                xfiboff=xfiboff, yfiboff=yfiboff, starflux=starflux)
                


def fit_iterate(data, guessflux, niter=10, psffun=SimplePSF,
                truth=None, verbose=False, useguess=True):
    nfib, nexp = data.shape
    if useguess:
        guess = initial_guess(data, guessflux, psffun)
    else:
        guess = dict(xfiboff=np.zeros(nfib, dtype='f8'),
                     yfiboff=np.zeros(nfib, dtype='f8'),
                     xtel=0,
                     ytel=0,
                     fwhm=1.,
                     transparency=1.)
    guessfib = [guessflux, guess['xfiboff'], guess['yfiboff']]
    guessim = [np.ones(nexp, dtype='f4')*guess['xtel'], 
               np.ones(nexp, dtype='f4')*guess['ytel'],
               np.ones(nexp, dtype='f4')*guess['fwhm'],
               np.ones(nexp, dtype='f4')*guess['transparency']]
    ndof = nfib*nexp-nexp*4-nfib*3+3
    # trailing +3 intended to account for perfect degeneracy between
    # changing fiber x, y, flux and compensating with image x, y, transparency

    if truth is not None:
        psf = [psffun(fwhm0) for fwhm0 in truth['fwhm']]
        fluxtrue = model_flux_full(data,
            truth['starflux'], truth['xfiboff'], truth['yfiboff'], 
            truth['xtel'], truth['ytel'], psf, truth['transparency'])
        chitrue = (data['spectroflux']-fluxtrue)*np.sqrt(data['spectroflux_ivar'])
        if verbose:
            print('True chi^2/dof: %f' % (np.sum(chitrue**2)/ndof))
            print('True chi^2/realdof: %f' % (np.sum(chitrue**2)/nfib/nexp))

    for i in range(niter):
        if verbose:
            print('Iteration %d' % i)
        impar, imunc, chiim = fit_images(
            data, *guessfib, guess=np.array(guessim).T, psffun=psffun)
        guessim = [impar[:, 0], impar[:, 1], impar[:, 2], impar[:, 3]]
        if verbose:
            print('chi2/dof: %f' % (np.sum(chiim**2)/ndof))
        if i == 0:
            # start the fiber parameter guesses in the bulk of the
            # dither offsets.
        #     print('new initial fiber guess enabled.')
        #     guessfib = [guessflux,
        #                 -np.median(data['delta_x_arcsec'], axis=1),
        #                 -np.median(data['delta_y_arcsec'], axis=1)]
            # start the fiber parameter guesses centered at the brightest
            # object
            brightestind = np.argmax(data['spectroflux'], axis=1)
            brightestdat = data[np.arange(nfib), brightestind]
            guessfib = [guessflux,
                        -(brightestdat['delta_x_arcsec'] +
                          impar[brightestind, 0]),
                        -(brightestdat['delta_y_arcsec'] +
                          impar[brightestind, 1])]
        fibpar, fibunc, chifib = fit_fibers(
            data, *guessim, guess=np.array(guessfib).T, psffun=psffun)
        # chifib: nfib x nim
        guessfib = [fibpar[:, 0], fibpar[:, 1], fibpar[:, 2]]
        if verbose:
            print('chi2/dof: %f' % (np.sum(chifib**2)/ndof))
    m = guessflux > 0
    zeropoint = np.median(guessfib[0][m]/guessflux[m])
    guessfib[0] /= zeropoint
    guessim[3] *= zeropoint
    chi2fib = np.sum(chifib**2, axis=1)
    chi2fibnull = np.sum(data['spectroflux']**2*data['spectroflux_ivar'],
                         axis=1)
    astodeg = 1/60/60
    cosdec = np.cos(np.radians(data['target_dec']))
    fiber_ditherfit_ra = ((data['delta_x_arcsec']+guessfib[1][:, None]+
                           guessim[0][None, :])*astodeg/cosdec +
                          data['target_ra'])
    fiber_ditherfit_dec = ((data['delta_y_arcsec']+guessfib[2][:, None]+
                            guessim[1][None, :])*astodeg +
                           data['target_dec'])

    return dict(xtel=guessim[0], dxtel=imunc[:, 0],
                ytel=guessim[1], dytel=imunc[:, 1],
                fwhm=guessim[2], dfwhm=imunc[:, 2],
                transparency=guessim[3], dtransparency=imunc[:, 3],
                starflux=guessfib[0], dstarflux=fibunc[:, 0],
                xfiboff=guessfib[1], dxfiboff=fibunc[:, 1],
                yfiboff=guessfib[2], dyfiboff=fibunc[:, 2],
                chi2fib=chi2fib, chi2fibnull=chi2fibnull,
                guessflux=guessflux, fiber=data['fiber'][:, 0],
                expid=data['expid'][0, :],
                fiber_ditherfit_ra=fiber_ditherfit_ra,
                fiber_ditherfit_dec=fiber_ditherfit_dec)


def test_performance(fluxguessaccuracy=0.2, verbose=False, niter=10, 
                     **kw):
    fakedata = fake_data(**kw)
    nfiber, nexp = fakedata['data'].shape
    fluxguess = fakedata['starflux']*(1+np.random.rand(nfiber)*
                                       fluxguessaccuracy)
    fitpar = fit_iterate(fakedata['data'], fluxguess, verbose=verbose,
                         niter=niter)
    plot_performance(fakedata, fitpar)
    return fakedata, fitpar


def test_patterns_scales(fluxguessaccuracy=0.2, verbose=False, niter=10,
                         scales=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2], 
                         patterns=['gaussian', 'box', 'rtheta', 'cross', 
                                   'telescope'],
                         seed=42, psffun=SimplePSF,
                         **kw):
    res = {}
    for pattern in patterns:
        for scale in scales:
            print(pattern, scale)
            fakedata = fake_data(seed=seed, pattern=pattern, ditherscale=scale,
                                 psffun=psffun, **kw)
            nfiber, nexp = fakedata['data'].shape
            fluxguess = fakedata['starflux']*(1+np.random.randn(nfiber)*
                                              fluxguessaccuracy)
            fitpar = fit_iterate(fakedata['data'], fluxguess, verbose=verbose,
                                 niter=niter, psffun=psffun, truth=fakedata)
            res[(pattern, scale)] = (fakedata, fitpar)
    return res


def plot_performance(fakedata, fitpar, rasterized=False):
    from matplotlib import pyplot as p
    fields = ['xfiboff', 'yfiboff', 'starflux', 'xtel', 'ytel', 'fwhm', 
              'transparency']
    for i, field in enumerate(fields):
        p.subplot(3, 3, i+1)
        alpha = 1 if i >= 3 else 0.05
        rast = True if rasterized and i >= 0 else False
        mm = [np.min(fakedata[field]), np.max(fakedata[field])]
        p.plot(mm, mm, linestyle='--', color='gray')
        p.plot(fakedata[field], fitpar[field], '+', alpha=alpha, 
               rasterized=True)
        p.xlim(mm)
        p.ylim(mm)
        p.xlabel('%s (sim)' % field)
        p.ylabel('%s (recovered)' % field)
        res = (fitpar[field]-fakedata[field])
        sigma = np.std(res)
        mean = np.mean(res)
        p.text(0.1, 0.9, '$\mu = %6.3f$' % mean, transform=p.gca().transAxes)
        p.text(0.1, 0.8, '$\sigma = %6.3f$' % sigma, 
               transform=p.gca().transAxes)
    

def convolve_gaussian_with_ellipse_direct(fwhm, aa, bb, pixscale=0.01):
    # do the simplest approach first
    # potential faster improvements:
    # Dustin's galaxy/pixelized model approach
    npixo2 = int(19//pixscale)//2
    xx, yy = np.meshgrid(np.arange(-npixo2, npixo2+1), 
                            np.arange(-npixo2, npixo2+1))
    xx = xx*pixscale
    yy = yy*pixscale
    pp = gaussian(xx, yy, fwhm)*pixscale**2
    ellipse = ((xx/aa)**2 + (yy/bb)**2 < 1).astype('f4')
    from scipy.signal import fftconvolve
    ppconv = fftconvolve(pp, ellipse[::-1, ::-1], mode='same')
    return ppconv


def convolve_gaussian_with_ellipse_airy(fwhm, aa, bb, pixscale=0.01):
    # the less simple approach
    # get the correct pixelized FFT of an ellipse of the appropriate size
    # multiply that with the FFT of the PSF image
    # FFT back to real space
    # saves errors in ellipse shape, trades an FFT for some Bessel function
    # calls.
    from numpy import fft
    from scipy.special import jv
    npixo2 = int(19//pixscale)//2
    xx, yy = np.meshgrid(np.arange(-npixo2, npixo2+1), 
                            np.arange(-npixo2, npixo2+1))
    xx = xx*pixscale
    yy = yy*pixscale
    pp = gaussian(xx, yy, fwhm)*pixscale**2
    Fpp = fft.rfft2(pp)
    freq1 = fft.fftfreq(pp.shape[0])
    freq2 = fft.rfftfreq(pp.shape[1])
    fperp = np.sqrt(freq2.reshape(1, -1)**2+(freq1*bb/aa).reshape(-1, 1)**2)
    scalefac = 2*np.pi*aa/pixscale
    Fellipse = 2*jv(1, fperp*scalefac)/(fperp*scalefac)
    Fellipse[0, 0] = 1.0
    Fellipse *= np.pi*aa*bb/pixscale/pixscale
    return fft.irfft2(Fpp*Fellipse, s=pp.shape)


class SimpleFiberIntegratedPSF(object):
    def __init__(self, fwhm, pixscale, psffun=gaussian):
        self.platescale = desimodel.io.load_platescale()
        self.fwhm = fwhm
        self.pixscale = pixscale
        self.psffun = psffun
        maxradfibrad = 107/2./np.min(self.platescale['radial_platescale'])
        maxazfibrad = 107/2./np.min(self.platescale['az_platescale'])
        npixx = np.floor(maxradfibrad / pixscale)
        npixy = np.floor(maxazfibrad / pixscale)
        x1 = np.arange(-npixx, npixx+1, 1)*pixscale
        y1 = np.arange(-npixy, npixy+1, 1)*pixscale
        xx, yy = np.meshgrid(x1, y1)
        inellipse = (xx/maxradfibrad)**2+(yy/maxazfibrad)**2 < 1**2
        self.xint = xx[inellipse]
        self.yint = yy[inellipse]

    def fiberfrac(self, x, y, dx, dy):
        scalar = np.isscalar(x)
        if scalar:
            x = np.atleast_1d(x)
            y = np.atleast_1d(y)
            dx = np.atleast_1d(dx)
            dy = np.atleast_1d(dy)
        rad = np.sqrt(x**2+y**2)
        theta = np.arctan2(y, x)
        # use real dx, dy for PSF
        res = self.psffun(
            dx[:, None]+self.xint[None, :], 
            dy[:, None]+self.yint[None, :], self.fwhm)
        radialrad = 107/2./np.interp(rad, self.platescale['radius'],
                                        self.platescale['radial_platescale'])
        azrad = 107/2./np.interp(rad, self.platescale['radius'],
                                    self.platescale['az_platescale'])
        xintr = (self.xint[None, :]*np.cos(theta[:, None])+
                 self.yint[None, :]*np.sin(theta[:, None]))
        yintr = (-self.xint[None, :]*np.sin(theta[:, None])+
                  self.yint[None, :]*np.cos(theta[:, None]))
        keep = ((xintr/radialrad[:, None])**2 + 
                (yintr/azrad[:, None])**2 < 1**2)
        res = np.sum(res*keep*self.pixscale**2, axis=1)
        return res

def read_dithering_sim(fn):
    sim = fits.getdata(fn)
    nfiber = len(sim)
    nexp = sim['fluxvariations'].shape[1]
    data = np.zeros((nfiber, nexp), dtype=[
        ('spectroflux', 'f4'), ('spectroflux_ivar', 'f4'),
        ('xfocal', 'f4'), ('yfocal', 'f4'),
        ('delta_x_arcsec', 'f4'), ('delta_y_arcsec', 'f4'),
        ('trueflux', 'f4')])
    data['spectroflux'] = sim['calc_signals_r']
    data['spectroflux_ivar'] = 1./(data['spectroflux']+40)
    # data['spectroflux'] += np.random.randn(data['spectroflux'].shape)*data['dflux']
    print('need to add noise; could improve this using calc_snr?')
    platescale = desimodel.io.load_platescale()
    rad = np.sqrt(sim['focal_x']**2+sim['focal_y']**2)[:, None]
    ang = np.arctan2(sim['focal_y'], sim['focal_x'])[:, None]
    raddither = (np.cos(ang)*sim['dither_pos_x'] + 
                 np.sin(ang)*sim['dither_pos_y'])
    angdither = (-np.sin(ang)*sim['dither_pos_x'] +
                 np.cos(ang)*sim['dither_pos_y'])
    radialps = np.interp(rad, platescale['radius'],
                            platescale['radial_platescale'])
    azps = np.interp(rad, platescale['radius'],
                        platescale['az_platescale'])
    xditheras = (np.cos(ang)*raddither/radialps +
                 -np.sin(ang)*angdither/azps)
    yditheras = (np.sin(ang)*raddither/radialps +
                 np.cos(ang)*angdither/azps)
    data['delta_x_arcsec'] = xditheras
    data['delta_y_arcsec'] = yditheras
    data['xfocal'] = sim['focal_x'][:, None]
    data['yfocal'] = sim['focal_y'][:, None]
    out = {}
    out['data'] = data
    out['starflux'] = np.sum(sim['f_in'][:, 644:1500], axis=1)*0.0014/0.4
    # f_in is full spectrum; 644:1500 corresponds roughly to Hbeta - Halpha
    # 486 - 656 A.  I hope!  overall scale factor is made up.  units might be
    # wrong (i.e., ergs vs. photons)
    out['xtel'] = np.zeros(nexp, dtype='f4')
    out['ytel'] = np.zeros(nexp, dtype='f4')
    out['xfiboff'] = sim['known_offset_x']
    out['yfiboff'] = sim['known_offset_y']
    out['transparency'] = np.ones(nexp, dtype='f4')
    return out


def read_dithering_sim_schlegel(fn):
    sim = fits.getdata(fn)
    nfiber, nexp = sim['spectroflux'].T.shape
    data = np.zeros((nfiber, nexp), dtype=[
        ('spectroflux', 'f4'), ('spectroflux_ivar', 'f4'),
        ('xfocal', 'f4'), ('yfocal', 'f4'),
        ('delta_x_arcsec', 'f4'), ('delta_y_arcsec', 'f4'),
        ('trueflux', 'f4')])
    data['spectroflux'] = sim['spectroflux'].T
    data['spectroflux_ivar'] = 1./(np.clip(sim['spectroflux'].T, 1, np.inf))
    data['delta_x_arcsec'] = sim['delta_x_arcsec'].T
    data['delta_y_arcsec'] = sim['delta_y_arcsec'].T
    data['xfocal'] = sim['xfocal'].T
    data['yfocal'] = sim['yfocal'].T
    out = {}
    out['data'] = data
    out['xtel'] = sim['truth_xteloffset']
    out['ytel'] = sim['truth_yteloffset']
    out['xfiboff'] = sim['truth_xfiboffset'][0, :]
    out['yfiboff'] = sim['truth_yfiboffset'][0, :]
    out['transparency'] = sim['truth_transparency']
    out['fwhm'] = sim['truth_fwhm']
    out['starflux'] = sim['flux_r'][0, :]
    return out


def fit_dithering_sims(directory, niter=10, psffun=None, verbose=True,
                       fluxguessaccuracy=0.2):
    fns = glob.glob(os.path.join(directory, '*/*/*.fits'))
    s = np.argsort(fns)
    fns = np.array(fns)[s]
    out = {}
    for i, fn in enumerate(fns):
        print(fn, '%d/%d' % (i, len(fns)))
        sim = read_dithering_sim(fn)
        nfiber = len(sim['starflux'])
        fluxguess = (sim['starflux']*(1+np.random.randn(nfiber))*
                     fluxguessaccuracy)
        fitpar = fit_iterate(sim['data'], fluxguess, verbose=verbose,
                             niter=niter, psffun=psffun)
        out[fn] = fitpar
    return out


def model_initial_guess(param, data=None, guessflux=None, psffun=None):
    xtel, ytel, axx, axy, ayx, ayy, fwhm, transparency = param
    scalefac = 1.0/400
    # 1" / 400 mm; xfiboff is in arcseconds, xfib is in mm
    # this defines a scaling so that 1" change over 1 focal plane radius
    # means a = 1
    xfocal, yfocal = data['xfocal'][:, 0]*scalefac, data['yfocal'][:, 0]*scalefac
    xfiboff = xtel + axx*xfocal + axy*yfocal
    yfiboff = ytel + ayx*xfocal + ayy*yfocal
    nexp = data.shape[1]
    psf = [psffun(fwhm) for i in range(nexp)]
    zero = np.zeros(1, dtype='f4')
    modflux = model_flux_full(data, guessflux, xfiboff, yfiboff, zero, zero, 
                              psf, transparency)
    return modflux
    

def chi_initial_guess(param, data=None, **kw):
    modflux = model_initial_guess(param, data=data, **kw)
    chi = (data['spectroflux']-modflux)*np.sqrt(data['spectroflux_ivar'])
    return chi.reshape(-1)
    

def initial_guess(data, guessflux, psffun, truth=None, epsfcn=1e-4, **kw):
    # initial guess
    # solve for: overall offset, scale, rotation, overal fwhm, transparency
    nparam = 2 + 4 + 2
    nfiber, nexp = data['spectroflux'].shape
    # xtel, ytel, axx, axy, ayx, ayy, fwhm, transparency
    guess = [0, 0, 0, 0, 0, 0, 1., 1.]
    args = dict(data=data, psffun=psffun, guessflux=guessflux)
    def chi(param, args=None):
        chid = damper(chi_initial_guess(param, **args), 5)
        return chid

    res = leastsq(chi, guess, args=(args,), epsfcn=epsfcn, **kw)
    chibest = chi(res[0], args=args)
    scalefac = 1.0/400
    xfocal, yfocal = data['xfocal'][:, 0]*scalefac, data['yfocal'][:, 0]*scalefac
    param = res[0]
    xfiboff = param[2]*xfocal + param[3]*yfocal
    yfiboff = param[4]*xfocal + param[5]*yfocal
    out = dict(xtel=param[0], ytel=param[1], axx=param[2], axy=param[3],
               ayx=param[4], ayy=param[5], fwhm=param[6], 
               transparency=param[7], xfiboff=xfiboff, yfiboff=yfiboff)
    return out


# def chi2andgrad(data, starflux, xfiboff, yfiboff, xtel, ytel, fwhm,
#                 transparency,
#                 psffun, hfwhm=1e-3, hoffset=1e-3):



def damper(chi, damp):
    """Pseudo-Huber loss function."""
    return 2*damp*np.sign(chi)*(np.sqrt(1+np.abs(chi)/damp)-1)
    # return chi/np.sqrt(1+np.abs(chi)/damp)


def damper_deriv(chi, damp, derivnum=1):
    """Derivative of the pseudo-Huber loss function."""
    if derivnum == 1:
        return (1+np.abs(chi)/damp)**(-0.5)
    if derivnum == 2:
        return -0.5*np.sign(chi)/damp*(1+np.abs(chi)/damp)**(-1.5)


def guess_starcounts(flux):
    # very roughly: spectroscopic flux = photometric flux
    # zeropoint = 26.8
    return flux*10.**((26.8-22.5)/2.5)


def quiver_plot_basic(xfocal, yfocal, xfiboff, yfiboff, **kw):
    from matplotlib import pyplot as p
    # -x since RA and X are in opposite directions!
    p.quiver(xfocal, yfocal, -xfiboff, yfiboff, scale=0.1, scale_units='x',
             **kw)
    p.gca().set_aspect('equal')
    p.xlabel('xfocal')
    p.ylabel('yfocal')
    p.text(0.1, 0.9, 'fiber offsets: 10 mm = 1"', transform=p.gca().transAxes)
    p.xlim(-450, 450)
    p.ylim(-450, 450)
    

def quiver_plot(sol, data):
    dchi2frac = (sol['chi2fibnull']-sol['chi2fib'])/sol['chi2fibnull']
    m = dchi2frac > 0.9
    from matplotlib import pyplot as p
    p.clf()
    p.gcf().set_size_inches(6, 5, forward=True)
    xoff = np.mean(sol['xfiboff'][m])
    yoff = np.mean(sol['yfiboff'][m])
    quiver_plot_basic(data['xfocal'][m, 0], data['yfocal'][m, 0],
                      sol['xfiboff'][m]-xoff,
                      sol['yfiboff'][m]-yoff)


def telparams_plot(sol, data):
    from matplotlib import pyplot as p
    p.clf()
    p.gcf().set_size_inches(6, 5, forward=True)
    meanxtel= np.mean(sol['xtel'])
    meanytel = np.mean(sol['ytel'])
    mjd = data['mjd_obs'][0, :]
    minmjd = np.floor(np.min(mjd))
    dmjd = mjd-minmjd
    p.plot(dmjd, sol['xtel']-meanxtel, '+', label='ra')
    p.plot(dmjd, sol['ytel']-meanytel, '+', label='dec')
    p.text(0.2, 0.9, 'Mean offset: %5.2f %5.2f' %
           (meanxtel, meanytel), transform=p.gca().transAxes)
    p.ylabel('offset (")')
    p.xlabel('MJD - %d' % minmjd)
    p.legend(loc='upper left')
    p.twinx()
    p.plot(dmjd, sol['fwhm'], 'rx', label='FWHM')
    p.plot(dmjd, sol['transparency'], 'bx', label='transparency')
    p.ylabel('FWHM, transparency')
    p.legend(loc='upper right')


def fiber_plot(sol, data, i, guess=None):
    from matplotlib import pyplot as p
    p.clf()
    p.gcf().set_size_inches(6, 5, forward=True)
    if guess is None:
        guess = 1
    else:
        guess = guess[i]
    dchi2frac = (sol['chi2fibnull']-sol['chi2fib'])/sol['chi2fibnull']
    good = dchi2frac > 0.9
    relflux = np.clip(data['spectroflux'][i, :], 1, np.inf)/guess
    p.scatter(data['delta_x_arcsec'][i, :]+sol['xtel'],
              data['delta_y_arcsec'][i, :]+sol['ytel'],
              c=-2.5*np.log10(relflux))
    p.plot(-sol['xfiboff'][i], -sol['yfiboff'][i], 'x')
    p.colorbar()
    p.xlim(-20, 20)
    p.ylim(-20, 20)
    p.gca().set_aspect('equal')
    extramessage = 'dchi2frac=%4.1f' % dchi2frac[i]
    if dchi2frac[i] < 0.9:
        extramessage = extramessage + ', EXCLUDED' 
    p.title(('Fiber %d' % data['fiber'][i, 0])+extramessage)


def prune_data(data, camera, snrcut=5, atleastnfib=20, atleastnim=5,
               snrbrightestcut=5, usepetals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    mgoodfib = np.all(np.isfinite(data[camera]['delta_x_arcsec']), axis=1)
    mgoodfib &= np.array([fib // 500 in usepetals
                          for fib in data[camera]['fiber'][:, 0]])
    mgoodim = ~np.all(~np.isfinite(data[camera]['delta_x_arcsec']), axis=0)
    tdata = {x: data[x][np.ix_(mgoodfib, mgoodim)] for x in data}
    for i in range(2):
        snr =  np.array([tdata[x]['spectroflux'] *
                         np.sqrt(tdata[x]['spectroflux_ivar']) for x in tdata])
        hassignal = snr > snrcut
        hassignal = np.sum(hassignal, axis=0) >= 2
        mgoodim = np.sum(hassignal, axis=0) >= atleastnfib
        mgoodfib = np.sum(hassignal, axis=1) >= atleastnim
        mgoodfib = mgoodfib & (np.max(snr, axis=(0, 2)) > snrbrightestcut)
        tdata = {x: tdata[x][np.ix_(mgoodfib, mgoodim)] for x in tdata}
    return tdata


def process(data, camera, outdir='.', label='dither%s',
            niter=10, overwrite=False, snrcut=5, atleastnfib=20,
            atleastnim=5, snrbrightestcut=5,
            usepetals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], **kw):
    from functools import partial
    tdata = prune_data(data, camera, snrcut=snrcut, atleastnfib=atleastnfib,
                       atleastnim=atleastnim, snrbrightestcut=snrbrightestcut,
                       usepetals=usepetals)
    tdata = tdata[camera]
    print('Attempting to solve for %d fibers in %d exposures.' % tdata.shape)
    print('Max snr: %5.1f' %
          np.max(tdata['spectroflux'] *
                 np.sqrt(tdata['spectroflux_ivar'])))
    guessbanddict = {'B':'g', 'R':'r', 'Z':'z'}
    guess = guess_starcounts(
        tdata['flux_'+guessbanddict[camera]][:, 0])
    psffun = partial(SimpleFiberIntegratedPSF, psffun=moffat, pixscale=0.2)
    sol = fit_iterate(tdata, guess, psffun=psffun, verbose=True, niter=niter,
                      **kw)
    nfib, nim = tdata.shape
    dtype = [(k, ('%d' % nim)+sol[k].dtype.descr[0][1]) for k in sol]
    out = np.zeros(nfib, dtype)
    for k in sol:
        if len(sol[k].shape) == 2:
            out[k] = sol[k]
        elif len(sol[k]) == out.shape[0]:
            out[k] = sol[k].reshape(-1, 1)
        else:
            out[k] = sol[k].reshape(1, -1)
    outfn = os.path.join(outdir, (label % camera)+'.fits')
    pdffn = os.path.join(outdir, (label % camera)+'.pdf')
    fits.writeto(outfn, out, overwrite=overwrite)
    dtype = [(k, ('%d' % nim)+tdata.dtype[k].descr[0][1])
             for k in tdata.dtype.fields]
    tdataout = np.zeros(nfib, dtype)
    for k in tdata.dtype.fields:
        tdataout[k] = tdata[k]
    fits.append(outfn, tdataout)
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as p
    pdf = PdfPages(pdffn)
    p.clf()
    quiver_plot(sol, tdata)
    p.title(label % camera)
    pdf.savefig()
    telparams_plot(sol, tdata)
    pdf.savefig()
    for i in range(tdata.shape[0]):
        p.clf()
        fiber_plot(sol, tdata, i, guess=guess)
        pdf.savefig()
    pdf.close()
    return out, sol, tdata


def multicolor_tel_plot(res):
    from matplotlib import pyplot as p
    p.clf()
    p.gcf().set_size_inches(6, 5, forward=True)
    fields = ['xtel', 'ytel', 'fwhm', 'transparency']
    solr = res['R'][1]
    solb = res['B'][1]
    solz = res['Z'][1]
    for i in range(4):
        p.subplot(2, 2, i+1)
        f = fields[i]
        p.plot(solr[f], solb[f], 'b+', label='B/R')
        p.plot(solr[f], solz[f], 'r+', label='Z/R')
        p.text(0.1, 0.9, f, transform=p.gca().transAxes)
        p.legend(loc='lower right')
        minmax = np.min(solr[f]), np.max(solr[f])
        p.plot(minmax, minmax, '--')


def multicolor_fiber_plot(res):
    from matplotlib import pyplot as p
    p.clf()
    p.gcf().set_size_inches(6, 5, forward=True)
    fields = ['xfiboff', 'yfiboff', 'starflux']
    solr = res['R'][1]
    solb = res['B'][1]
    solz = res['Z'][1]
    for i in range(3):
        p.subplot(2, 2, i+1)
        f = fields[i]
        rr = solr[f]
        bb = solb[f]
        zz = solz[f]
        if f == 'starflux':
            rr = -2.5*np.log10(np.clip(rr, 1, np.inf)/solr['guessflux'])
            bb = -2.5*np.log10(np.clip(bb, 1, np.inf)/solb['guessflux'])
            zz = -2.5*np.log10(np.clip(zz, 1, np.inf)/solz['guessflux'])
        p.plot(rr, bb, 'b+', label='B/R')
        p.plot(rr, zz, 'r+', label='Z/R')
        p.text(0.1, 0.9, f, transform=p.gca().transAxes)
        p.legend(loc='lower right')
        minmax = (np.min(rr), np.max(rr))
        p.plot(minmax, minmax, '--')


def process_all(data, outdir='.', label='dither%s',
                usepetals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                niter=10, overwrite=False, **kw):
    from copy import deepcopy
    data = deepcopy(data)
    for c in data:
        m = np.isfinite(data[c]['spectroflux_ivar'])
        m[m] = data[c]['spectroflux_ivar'][m] > 0
        data[c]['spectroflux_ivar'][m] = (
            data[c]['spectroflux_ivar'][m]**(-1) +
            (0.01*data[c]['spectroflux'][m])**2)**(-1)
    res = {}
    for c in data:
        res[c] = process(data, c, outdir=outdir, label=label,
                         usepetals=usepetals, niter=niter,
                         overwrite=overwrite, **kw)
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as p
    pdffn = os.path.join(outdir, (label % 'BRZ')+'.pdf')
    pdf = PdfPages(pdffn)
    p.clf()
    p.title(label % 'BRZ')
    multicolor_tel_plot(res)
    pdf.savefig()
    multicolor_fiber_plot(res)
    pdf.savefig()
    pdf.close()
    return res


def plot_one_exp(data, expid, usepetals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 snrcut=100, xrange=[-15, 15], yrange=[-15, 15]):
    from matplotlib import pyplot as p
    guessbanddict = {'B':'g', 'R':'r', 'Z':'z', 'A':'r'}
    newdata = {}
    for key in data:
        newdata[key] = data[key]
    data = newdata
    data['A'] = data['R']
    data['A']['spectroflux'] = np.median(np.array([
        data[c]['spectroflux'] for c in 'BRZ']), axis=0)
    data['A']['spectroflux_ivar'] = np.median(np.array([
        data[c]['spectroflux_ivar'] for c in 'BRZ']), axis=0)
    for i, camera in enumerate('BRZA'):
        tdata = data[camera]
        ind = np.flatnonzero(tdata['expid'][0, :] == expid)[0]
        guess = guess_starcounts(
            tdata['flux_'+guessbanddict[camera]][:, 0])
        p.subplot(2, 4, i+1)
        relflux = -2.5*np.log10(np.clip(tdata['spectroflux'], 1, np.inf)/
                                np.clip(guess, 1, np.inf).reshape(-1, 1))
        snr = tdata['spectroflux']*np.sqrt(tdata['spectroflux_ivar'])
        good = ((snr[:, ind] > snrcut) & (relflux[:, ind] < 5) &
                np.isfinite(relflux[:, ind]) & 
                np.isfinite(tdata['delta_x_arcsec'][:, ind]))
        m = np.array([tpet in usepetals
                      for tpet in tdata['fiber'][:, ind] // 500])
        sz = 0.05+5*good
        p.scatter(tdata['xfocal'][~good & m, ind],
                  tdata['yfocal'][~good & m, ind],
                  c=relflux[~good & m, ind], s=sz[~good & m], vmax=5, vmin=0,
                  rasterized=True)
        p.scatter(tdata['xfocal'][good & m, ind],
                  tdata['yfocal'][good & m, ind],
                  c=relflux[good & m, ind], s=sz[good & m], vmax=5, vmin=0,
                  rasterized=True)
        p.title(camera)
        p.suptitle('EXPID: %d' % expid)
        p.gca().set_aspect('equal')
        p.xlim(-450, 450)
        p.ylim(-450, 450)
        p.subplot(2, 4, i+5)
        p.scatter(tdata['delta_x_arcsec'][~good & m, ind],
                  tdata['delta_y_arcsec'][~good & m, ind],
                  c=relflux[~good & m, ind], s=sz[~good & m], vmax=5, vmin=0,
                  rasterized=True)
        p.scatter(tdata['delta_x_arcsec'][good & m, ind],
                  tdata['delta_y_arcsec'][good & m, ind],
                  c=relflux[good & m, ind], s=sz[good & m], vmax=5, vmin=0,
                  rasterized=True)
        p.gca().set_aspect('equal')
        p.xlim(*xrange)
        p.ylim(*yrange)

    
def plot_sequence(sequence, fn, **kw):
    import ditherdata
    res = ditherdata.rearrange_table(sequence._exposure_table)
    exps = res['R']['expid'][0, :]
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(fn)
    from matplotlib import pyplot as p
    p.figure()
    p.gcf().set_size_inches(14, 7, forward=True)
    for i, exp in enumerate(exps):
        print(i, len(exps))
        p.clf()
        plot_one_exp(res, exp, **kw)
        pdf.savefig()
    pdf.close()
        

def model_rotation_offset(param, data=None, guessflux=None, psffun=None,
                          fwhm=None, transparency=None):
    xtel, ytel, theta = param
    thetarad = np.radians(theta)
    # convert to microns; divide by plate scale (roughly!)
    scalefac = 1000 / 70
    xfocal, yfocal = data['xfocal']*scalefac, data['yfocal']*scalefac
    # off diagonal terms have the same sign because, confusingly,
    # xfiboff is in RA and xfocal is in CS5, and these increase in
    # different directions.
    xfiboff = xtel + (np.cos(thetarad)-1)*xfocal + np.sin(thetarad)*yfocal
    yfiboff = ytel + np.sin(thetarad)*xfocal + (np.cos(thetarad)-1)*yfocal
    nexp = data.shape[1]
    psf = [psffun(fwhm) for i in range(nexp)]
    zero = np.zeros(1, dtype='f4')
    transparency = np.array(transparency)
    modflux = model_flux_full(data, guessflux, xfiboff, yfiboff, zero, zero, 
                              psf, transparency)
    return modflux
    

def chi_rotation_offset(param, data=None, **kw):
    modflux = model_rotation_offset(param, data=data, **kw)
    scale = (np.sum(modflux*data['spectroflux']*data['spectroflux_ivar'])/
             np.sum(modflux**2*data['spectroflux_ivar']))
    scale = 1
    chi = (data['spectroflux']-scale*modflux)*np.sqrt(data['spectroflux_ivar'])
    return chi.reshape(-1)
    

def rotation_offset(data, camera, guessflux=None, psffun=None,
                    epsfcn=1e-4, fwhm=1.5,
                    atleastnfib=1, atleastnim=1,
                    snrbrightestcut=5, snrcut=5, usepetals=list(range(10)),
                    index=-1,
                    **kw):
    # solve for: overall offset, rotation, overal fwhm
    nparam = 2 + 1 + 2
    if index != -1:
        newdata = {c:data[c][:, index:index+1].copy() for c in data}
        data = newdata
    data = prune_data(data, camera, snrcut=snrcut, atleastnfib=atleastnfib,
                      atleastnim=atleastnim, snrbrightestcut=snrbrightestcut,
                      usepetals=usepetals)
    data = data[camera]
    # Lots of objects have SNR ~ 5, presumably ~spuriously in this exposure.
    # force everything with SNR < 5 to have flux zero.
    # data['spectroflux'][data['spectroflux']*
    #                     np.sqrt(data['spectroflux_ivar']) < 5] = 0.
    nfiber, nexp = data['spectroflux'].shape
                
    if guessflux is None:
        guessbanddict = {'B':'g', 'R':'r', 'Z':'z', 'A':'r'}
        guessflux = guess_starcounts(
            data['flux_'+guessbanddict[camera]][:, 0])
    if psffun is None:
        from functools import partial
        # psffun = partial(SimpleFiberIntegratedPSF, psffun=moffat,
        #                  pixscale=0.2)
        psffun = partial(SimplePSF, psffun=moffat)
    args = dict(data=data, psffun=psffun, guessflux=guessflux,
                fwhm=fwhm, transparency=1)
    def chi(param, args=None):
        dchi = damper(chi_rotation_offset(param, **args), 5)
        return dchi

    bestchi2 = np.inf
    # apparently this problem is nasty and has a lot of local minima?  Start
    # with a grid search?
    for dx in np.linspace(-20, 20, 41):
        for dy in np.linspace(-20, 20, 41):
            for rot in np.linspace(-0.2, 0.2, 41):
                chi2 = np.sum(chi([dx, dy, rot], args=args)**2)
                if chi2 < bestchi2:
                    bestchi2 = chi2
                    bestparam = [dx, dy, rot]
    guess = bestparam

    res = leastsq(chi, guess, args=(args,), epsfcn=epsfcn, **kw)
    chibest = chi(res[0], args=args)
    param = res[0]
    theta = np.radians(param[2])
    xfocal, yfocal = data['xfocal'][:, 0], data['yfocal'][:, 0]
    # off diagonal terms have the same sign because, confusingly,
    # xfiboff is in RA and xfocal is in CS5, and these increase in
    # different directions.
    xfiboff = (np.cos(theta)-1)*xfocal + np.sin(theta)*yfocal
    yfiboff = np.sin(theta)*xfocal + (np.cos(theta)-1)*yfocal
    xfiboff, yfiboff = xfiboff*1000/70, yfiboff*1000/70
    out = dict(xtel=param[0], ytel=param[1], rotation=param[2],
               xfiboff=xfiboff, yfiboff=yfiboff)
    return out

