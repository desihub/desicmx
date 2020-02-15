#!/usr/bin/env python

import os,sys
import fitsio
import astropy.io.fits as pyfits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

from desimeter.transform.radec2tan import hadec2xy
from desimeter.simplecorr import SimpleCorr

import argparse
parser = argparse.ArgumentParser(description='Compare desimeter & dither fiber positions')
parser.add_argument('dither_file', type=str, help='dither file name')
parser.add_argument('fvc_file', type=str, help='desi_fvc_proc file name')
parser.add_argument('fiberassign_file', type=str, help='fiberassign file name')
parser.add_argument('-e', '--expid', type=int, default=None, help='expid to use from dither file')
args = parser.parse_args()


quiver_units="width"
quiver_scale=20.
def arrow() :
    xarrow=-0.025
    dxarrow=2.#arcsec
    plt.quiver(xarrow,-0.025,dxarrow,0.,color="black",units=quiver_units,scale=quiver_scale)
    plt.text(xarrow,-0.029,"{} arcsec".format(dxarrow))

def text(blabla) :
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.029,-0.029,blabla,fontsize=8, bbox=props,verticalalignment='bottom', horizontalalignment='right')


dither_file = args.dither_file
fvc_file = args.fvc_file
fiberassign_file = args.fiberassign_file

# Eddie's dithers
t=fitsio.read(dither_file)
print(t.dtype.names)
if args.expid is None:
    i=0
else:
    ind = np.flatnonzero(t['expid'][0, :] == args.expid)
    if len(ind) > 0:
        i = ind[0]
    else:
        raise ValueError(f'could not find expid {args.expid} in dither file.')

err=np.sqrt(t['dxfiboff'][:,i]**2+t['dyfiboff'][:,i]**2)
roff=np.sqrt(t['xfiboff'][:,i]**2+t['yfiboff'][:,i]**2)
jj=np.where((err<0.02)&(roff<4.))[0]

dither_ra    = t['fiber_ditherfit_ra'][jj,i]
dither_dec   = t['fiber_ditherfit_dec'][jj,i]
dither_fiber = t['fiber'][jj,i].astype(int)

# desimeter RA Dec
t=Table.read(fvc_file)
ii=(t["RA"]!=0)
t=t[:][ii]
desimeter_ra  = t["RA"]
desimeter_dec = t["DEC"]

# fiberassign
f=Table.read(fiberassign_file)
head = fitsio.read_header(fiberassign_file)
dico = { loc : i for i,loc in enumerate(f["LOCATION"])}
target_ra=np.zeros(len(t))
target_dec=np.zeros(len(t))
target_fiber=np.zeros(len(t))
for j,loc in enumerate(t["LOCATION"]) :
    if loc in dico :
        i=dico[loc]
        target_ra[j] = f["TARGET_RA"][i]
        target_dec[j] = f["TARGET_DEC"][i]
        target_fiber[j] = f["FIBER"][i]

tel_ra=head["REQRA"]
tel_dec=head["REQDEC"]
desimeter_x,desimeter_y=hadec2xy(-desimeter_ra+tel_ra,desimeter_dec,0,tel_dec)
target_x,target_y=hadec2xy(-target_ra+tel_ra,target_dec,0,tel_dec)
dither_x,dither_y=hadec2xy(-dither_ra+tel_ra,dither_dec,0,tel_dec)

# Now find match to dithers
dither_index = np.zeros(dither_fiber.size,dtype=int)
for i,fiber in enumerate(dither_fiber) :
    jj=np.where(target_fiber == fiber)[0]
    if jj.size ==  1 :
        dither_index[i] = jj[0]
    else :
        dither_index[i] = -1

desimeter_x  = desimeter_x[dither_index[dither_index>=0]]
desimeter_y  = desimeter_y[dither_index[dither_index>=0]]
target_x  = target_x[dither_index[dither_index>=0]]
target_y  = target_y[dither_index[dither_index>=0]]
dither_x = dither_x[dither_index>=0]
dither_y = dither_y[dither_index>=0]

rad2arcsec = 180.*3600/np.pi
desimeter_dx=(desimeter_x-target_x)*rad2arcsec # now arcsec
desimeter_dy=(desimeter_y-target_y)*rad2arcsec # now arcsec
dither_dx=(dither_x-target_x)*rad2arcsec
dither_dy=(dither_y-target_y)*rad2arcsec


ddx = desimeter_dx-dither_dx
ddy = desimeter_dy-dither_dy
d=np.sqrt(ddx**2+ddy**2) # arcsec
ii=(d<10)

desimeter_dx = desimeter_dx[ii]
desimeter_dy = desimeter_dy[ii]
desimeter_x  = desimeter_x[ii]
desimeter_y  = desimeter_y[ii]
target_x  = target_x[ii]
target_y  = target_y[ii]
dither_dx = dither_dx[ii]
dither_dy = dither_dy[ii]
dither_x = dither_x[ii]
dither_y = dither_y[ii]
ddx = ddx[ii]
ddy = ddy[ii]

# fit a transform to adjust desimeter to the dither results
corr=SimpleCorr()
corr.fit(desimeter_x,desimeter_y,dither_x,dither_y)
desimeter_x_bis , desimeter_y_bis = corr.apply(desimeter_x,desimeter_y)
desimeter_dx_bis=(desimeter_x_bis-target_x)*rad2arcsec # now arcsec
desimeter_dy_bis=(desimeter_y_bis-target_y)*rad2arcsec # now arcsec
ddx_bis = desimeter_dx_bis-dither_dx
ddy_bis = desimeter_dy_bis-dither_dy

title=os.path.basename(dither_file).replace(".fits","-vs-desimeter")
plt.figure(title)



a = plt.subplot(3,2,1)
a.set_title("dither")
plt.quiver(target_x,target_y,dither_dx,dither_dy,color="red",units=quiver_units,scale=quiver_scale,label="dither")
arrow()
rms=np.sqrt(np.mean(dither_dx**2+dither_dy**2))
text("rms = {:3.2f}''".format(rms))

a = plt.subplot(3,2,2)
#a.set_title("desimeter -> dither")
#plt.plot([0,1],[0,1],color="white")
plt.axis('off')
rad2arcsec = 180*3600/np.pi
blabla = "tranformation\n"
blabla += "desimeter -> dither\n\n"
blabla +=  "dx  = {:3.2f}''\n".format(corr.dx*rad2arcsec)
blabla += "dy  = {:3.2f}''\n".format(corr.dy*rad2arcsec)
blabla += "sxx = {:6.5f}\n".format(corr.sxx-1)
blabla += "syy = {:6.5f}\n".format(corr.syy-1)
blabla += "sxy = {:6.5f}\n".format(corr.sxy)
blabla += "rot = {:3.2f}''\n".format(corr.rot_deg*3600)
blabla += "rms = {:3.2f}''\n".format(corr.rms*180*3600/np.pi)
plt.text(0,0,blabla,fontsize=12)

a = plt.subplot(3,2,3)
a.set_title("desimeter")
plt.quiver(target_x,target_y,desimeter_dx,desimeter_dy,color="red",units=quiver_units,scale=quiver_scale,label="dither")
arrow()
rms=np.sqrt(np.mean(desimeter_dx**2+desimeter_dy**2))
text("rms = {:3.2f}''".format(rms))



a = plt.subplot(3,2,4)
a.set_title("desimeter(corr)")
plt.quiver(target_x,target_y,desimeter_dx_bis,desimeter_dy_bis,color="red",units=quiver_units,scale=quiver_scale,label="dither")
arrow()
rms=np.sqrt(np.mean(desimeter_dx_bis**2+desimeter_dy_bis**2))
text("rms = {:3.2f}''".format(rms))

a = plt.subplot(3,2,5)
a.set_title("desimeter-dither")
plt.quiver(target_x,target_y,desimeter_dx-dither_dx,desimeter_dy-dither_dy,color="red",units=quiver_units,scale=quiver_scale,label="dither")
arrow()
rms=np.sqrt(np.mean((desimeter_dx-dither_dx)**2+(desimeter_dy-dither_dy)**2))
text("rms = {:3.2f}''".format(rms))

a = plt.subplot(3,2,6)
a.set_title("desimeter(corr)-dither")
plt.quiver(target_x,target_y,desimeter_dx_bis-dither_dx,desimeter_dy_bis-dither_dy,color="red",units=quiver_units,scale=quiver_scale,label="dither")
arrow()
rms=np.sqrt(np.mean((desimeter_dx_bis-dither_dx)**2+(desimeter_dy_bis-dither_dy)**2))
text("rms = {:3.2f}''".format(rms))



plt.show()




rms = np.sqrt(np.mean(ddx**2+ddy**2))
text = "rms desimeter = {:3.2f}''\n".format(rms)
rms = np.sqrt(np.mean(dither_dx**2+dither_dy**2))
text += "rms dither(platemaker) = {:3.2f}''\n".format(rms)

plt.xlabel("x_tan")
plt.ylabel("y_tan")
plt.legend(loc="upper left")

plt.subplot(3,2,2)
plt.quiver(target_x,target_y,desimeter_dx_bis,desimeter_dy_bis,color="gray",alpha=0.8,units=quiver_units,scale=quiver_scale,label="desimeter (corr)")
plt.quiver(target_x,target_y,dither_dx,dither_dy,color="red",units=quiver_units,scale=quiver_scale,label="dither")
plt.quiver(xarrow,-0.025,dxarrow,0.,color="black",units=quiver_units,scale=quiver_scale)
plt.text(xarrow,-0.029,"{} arcsec".format(dxarrow))



plt.xlabel("x_tan")
plt.ylabel("y_tan")
plt.legend(loc="upper left")
plt.subplot(3,2,1)
plt.quiver(target_x,target_y,desimeter_dx,desimeter_dy,color="grey",alpha=0.8,units=quiver_units,scale=quiver_scale,label="desimeter (uncorr)")
plt.quiver(target_x,target_y,dither_dx,dither_dy,color="red",units=quiver_units,scale=quiver_scale,label="dither")

xarrow=-0.025
dxarrow=2.#arcsec
plt.quiver(xarrow,-0.025,dxarrow,0.,color="black",units=quiver_units,scale=quiver_scale)
plt.text(xarrow,-0.029,"{} arcsec".format(dxarrow))

rms = np.sqrt(np.mean(ddx**2+ddy**2))
text = "rms desimeter = {:3.2f}''\n".format(rms)
rms = np.sqrt(np.mean(dither_dx**2+dither_dy**2))
text += "rms dither(platemaker) = {:3.2f}''\n".format(rms)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.03,-0.03,text,fontsize=8, bbox=props,verticalalignment='bottom', horizontalalignment='right')

plt.xlabel("x_tan")
plt.ylabel("y_tan")
plt.legend(loc="upper left")

plt.subplot(3,2,2)
plt.quiver(target_x,target_y,desimeter_dx_bis,desimeter_dy_bis,color="gray",alpha=0.8,units=quiver_units,scale=quiver_scale,label="desimeter (corr)")
plt.quiver(target_x,target_y,dither_dx,dither_dy,color="red",units=quiver_units,scale=quiver_scale,label="dither")
plt.quiver(xarrow,-0.025,dxarrow,0.,color="black",units=quiver_units,scale=quiver_scale)
plt.text(xarrow,-0.029,"{} arcsec".format(dxarrow))

rad2arcsec = 180*3600/np.pi
text =  "dx  = {:3.2f}''\n".format(corr.dx*rad2arcsec)
text += "dy  = {:3.2f}''\n".format(corr.dy*rad2arcsec)
text += "sxx = {:6.5f}\n".format(corr.sxx-1)
text += "syy = {:6.5f}\n".format(corr.syy-1)
text += "sxy = {:6.5f}\n".format(corr.sxy)
text += "rot = {:3.2f}''\n".format(corr.rot_deg*3600)
text += "rms = {:3.2f}''\n".format(corr.rms*180*3600/np.pi)
#text += "rms_dither= {:3.2f}''\n".format(rms_dither)

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.03,-0.03,text,fontsize=8, bbox=props,verticalalignment='bottom', horizontalalignment='right')


plt.xlabel("x_tan")
plt.ylabel("y_tan")
plt.legend(loc="upper left")

plt.subplot(2,2,3)
plt.quiver(target_x,target_y,ddx,ddy,color="k",units=quiver_units,scale=quiver_scale,label="desimeter - dither")

rms = np.sqrt(np.mean(ddx**2+ddy**2))
text = "rms = {:3.2f}''\n".format(rms)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.03,-0.03,text,fontsize=8, bbox=props,verticalalignment='bottom', horizontalalignment='right')

plt.xlabel("x_tan")
plt.ylabel("y_tan")
plt.legend(loc="upper left")

plt.subplot(2,2,4)
plt.quiver(target_x,target_y,ddx_bis,ddy_bis,color="k",units=quiver_units,scale=quiver_scale,label="desimeter (corr) - dither")

rms = np.sqrt(np.mean(ddx_bis**2+ddy_bis**2))
text = "rms = {:3.2f}''\n".format(rms)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.03,-0.03,text,fontsize=8, bbox=props,verticalalignment='bottom', horizontalalignment='right')

plt.xlabel("x_tan")
plt.ylabel("y_tan")
plt.legend(loc="upper left")


plt.show()




plt.show()
