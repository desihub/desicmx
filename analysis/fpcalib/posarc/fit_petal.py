#!/usr/bin/env python

import sys
import os.path
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

filenames = sys.argv[1:]

x2i=None
y2i=None
loci=None
for filename in filenames :
    t=Table.read(filename)
    #print(t.dtype.names)
    x1=t["X_FP_METRO"]
    y1=t["Y_FP_METRO"]
    x2=t["X_FP_THETA_AXIS"]
    y2=t["Y_FP_THETA_AXIS"]
    loc=t["LOCATION"]
    if x2i is None :
        x2i=x2
        y2i=y2
        loci=loc
        filenamei=filename
    else :
        print("repeatability")
        print("--------------------------")
        dx=[]
        dy=[]
        xx=[]
        yy=[]
        for j,l in enumerate(loci) :
            ii=np.where(loc==l)[0]
            if ii.size == 1 :
                i=ii[0]
                xx.append(x2i[j])
                yy.append(y2i[j])
                dx.append(x2[i]-x2i[j])
                dy.append(y2[i]-y2i[j])

        dx = np.array(dx)
        dy = np.array(dy)

        mdx=np.mean(dx)*1000
        mdy=np.mean(dy)*1000
        rmsdx=np.std(dx)*1000
        rmsdy=np.std(dy)*1000

        dr2 = (1000*dx-mdx)**2 + (1000*dy-mdy)**2
        rmsdr = np.sqrt(np.mean(dr2))
        
        print("mean x = {:3.2f} um".format(mdx))
        print("mean y = {:3.2f} um".format(mdy))
        print("rms x = {:3.1f} um".format(rmsdx))
        print("rms y = {:3.1f} um".format(rmsdy))
        print("rms r = {:3.1f} um".format(rmsdr))
        print("")

        title = 'Repeatability\n{} - {}'.format(
                os.path.basename(filename), os.path.basename(filenamei))
        plt.figure(title, figsize=(6.1,6))
        a=plt.subplot(1,1,1)
        a.set_title(title)
        a.quiver(xx,yy,dx,dy, scale=1/2000, scale_units='xy')
        a.set_xlabel("DX_FP (mm)")
        a.set_ylabel("DX_FP (mm)")

    title=os.path.basename(filename).split(".")[0]
    plt.figure(title, figsize=(6.1,6))
    a=plt.subplot(1,1,1)
    a.set_title(title)
    a.quiver(x1,y1,x2-x1,y2-y1)
    a.set_xlabel("X_FP (mm)")
    a.set_ylabel("X_FP (mm)")
    
    
    print(filename)
    print("--------------------------")
    for petal in range(10) :

        ii=np.where(t["LOCATION"]//1000 == petal)[0]
        dx=x2[ii]-x1[ii]
        dy=y2[ii]-y1[ii]
        mdx=np.mean(dx)*1000
        mdy=np.mean(dy)*1000
        rmsx=np.std(dx)*1000
        rmsy=np.std(dy)*1000
        edx=rmsx/np.sqrt(dx.size-1)
        edy=rmsy/np.sqrt(dy.size-1)
        print("petal {} dx= {:3.0f} +- {:3.1f} dy= {:3.0f} +- {:3.1f} rmsx= {:3.0f} rmsy= {:3.0f}".format(petal,mdx,edx,mdy,edy,rmsx,rmsy))
    print("")

plt.show()    
