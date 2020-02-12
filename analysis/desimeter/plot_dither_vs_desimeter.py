#!/usr/bin/env python

import os,sys
import fitsio
import astropy.io.fits as pyfits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

from desimeter.transform.radec2tan import hadec2xy
from desimeter.simplecorr import SimpleCorr

dither_file = sys.argv[1]
fvc_file = sys.argv[2]
fiberassign_file = sys.argv[3]


t=fitsio.read(dither_file)
print(t.dtype.names)
i=0
err=np.sqrt(t['dxfiboff'][:,i]**2+t['dyfiboff'][:,i]**2)
roff=np.sqrt(t['xfiboff'][:,i]**2+t['yfiboff'][:,i]**2)
jj=np.where((err<0.02)&(roff<4.))[0]
dither_ra  = t['fiber_ditherfit_ra'][jj,i]
dither_dec = t['fiber_ditherfit_dec'][jj,i]

# Eddie: xfiboff = PM RA of star - true RA of star
# opposite sign because xtan goes as -RA
dither_dx  = -t['xfiboff'][jj,i]
dither_dy  = t['yfiboff'][jj,i]

dither_fiber = t['fiber'][jj,i].astype(int)

t=Table.read(fvc_file)
ii=(t["RA"]!=0)
t=t[:][ii]
desimeter_ra=t["RA"]
desimeter_dec=t["DEC"]

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
x1,y1=hadec2xy(-desimeter_ra+tel_ra,desimeter_dec,0,tel_dec)
x2,y2=hadec2xy(-target_ra+tel_ra,target_dec,0,tel_dec)
desimeter_dx=(x1-x2)*180.*3600/np.pi # now arcsec
desimeter_dy=(y1-y2)*180.*3600/np.pi # now arcsec

dither_x,dither_y=hadec2xy(-dither_ra+tel_ra,dither_dec,0,tel_dec)

# Now find match to dithers
dither_index = np.zeros(dither_fiber.size,dtype=int)
for i,fiber in enumerate(dither_fiber) :
    jj=np.where(target_fiber == fiber)[0]
    if jj.size ==  1 :
        dither_index[i] = jj[0]
    else :
        dither_index[i] = -1

d=np.sqrt(desimeter_dx**2+desimeter_dy**2) # arcsec
ii=(target_ra!=0)&(d<10)


corr=SimpleCorr()
if 0 : # fit to fiber assign
    corr.fit(x1[ii],y1[ii],x2[ii],y2[ii])
    print(corr)
else : # fit to dither test
    ii1=dither_index[dither_index>=0]
    ii2=(dither_index>=0)
    corr.fit(x1[ii1],y1[ii1],(dither_x+dither_dx*np.pi/(180.*3600.))[ii2],(dither_y+dither_dy*np.pi/(180.*3600.))[ii2])
    print(corr)
    

x3,y3 = corr.apply(x1,y1)
desimeter_dx_bis=(x3-x2)*180.*3600/np.pi # now arcsec
desimeter_dy_bis=(y3-y2)*180.*3600/np.pi # now arcsec

dist=np.sqrt(desimeter_dx_bis**2+desimeter_dy_bis**2) # arcsec
ii=(target_ra!=0)&(dist<3)

target_ra  = target_ra[ii]
target_dec = target_dec[ii]
desimeter_ra  = desimeter_ra[ii]
desimeter_dec = desimeter_dec[ii]
desimeter_x  = x1[ii]
desimeter_y  = y1[ii]
desimeter_dx  = desimeter_dx[ii]
desimeter_dy  = desimeter_dy[ii]
desimeter_dx_bis  = desimeter_dx_bis[ii]
desimeter_dy_bis  = desimeter_dy_bis[ii]

print(np.std(desimeter_dx))
print(np.std(dither_dx))

title=os.path.basename(dither_file).replace(".fits","-vs-desimeter")
plt.figure(title)

scale=20.
plt.subplot(121)
plt.quiver(desimeter_x,desimeter_y,desimeter_dx,desimeter_dy,color="grey",alpha=0.8,units="width",scale=scale,label="desimeter (uncorr)")
plt.quiver(dither_x,dither_y,dither_dx,dither_dy,color="red",units="width",scale=scale,label="dither")
x=-0.03
dx=2
plt.quiver(x,-0.025,dx,0.,color="black",units="width",scale=scale)
plt.text(x,-0.029,"{} arcsec".format(dx))
plt.xlabel("x_tan")
plt.ylabel("y_tan")
plt.legend(loc="upper left")

plt.subplot(122)
plt.quiver(desimeter_x,desimeter_y,desimeter_dx_bis,desimeter_dy_bis,color="gray",alpha=0.8,units="width",scale=scale,label="desimeter (corr)")
plt.quiver(dither_x,dither_y,dither_dx,dither_dy,color="red",units="width",scale=scale,label="dither")
plt.quiver(x,-0.025,dx,0.,color="black",units="width",scale=scale)
plt.text(x,-0.029,"{} arcsec".format(dx))

rad2arcsec = 180*3600/np.pi
text =  "dx  = {:3.2f}''\n".format(corr.dx*rad2arcsec)
text += "dy  = {:3.2f}''\n".format(corr.dy*rad2arcsec)
text += "sxx = {:6.5f}\n".format(corr.sxx-1)
text += "syy = {:6.5f}\n".format(corr.syy-1)
text += "sxy = {:6.5f}\n".format(corr.sxy)
text += "rot = {:3.2f}''\n".format(corr.rot_deg*3600)

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.03,-0.03,text,fontsize=8, bbox=props,verticalalignment='bottom', horizontalalignment='right')


plt.xlabel("x_tan")
plt.ylabel("y_tan")
plt.legend(loc="upper left")

plt.show()




plt.show()
