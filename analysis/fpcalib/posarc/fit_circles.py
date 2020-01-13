#!/usr/bin/env python

import sys
from astropy.table import Table
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import argparse

# from desimeter.transform.xy2qs import xy2qs, qs2xy
from desimodel.focalplane import xy2qs, qs2xy

#- Transform between CS5 x,y and curved focal surface
def xy2uv(x, y):
    q, s = xy2qs(x, y)
    qrad = np.radians(q)
    u = s*np.cos(qrad)
    v = s*np.sin(qrad)
    return u, v

def uv2xy(u, v):
    s = np.sqrt(u**2 + v**2)
    qrad = np.arctan2(v, u)
    q = np.degrees(qrad)
    x, y = qs2xy(q, s)
    return x, y

#- Least squares fit to circle; adapted from "Method 2b" in
#- https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

def fit_circle(xin, yin):

    #- Transform to curved focal surface which is closer to a real circle
    x, y = xy2uv(xin, yin)
    x_m, y_m, r = fast_fit_circle(x, y)
    
    #- If r is too small or too big then either this positioner wasn't moving
    #- or the points are mismatched for a bad fit.
    if (r < 1.0) or (r > 5.0):
        raise ValueError('Bad circle fit')

    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc     = c
        df2b_dc    = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
        df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
        df2b_dc    = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    
    center_estimate = x_m, y_m
    center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)

    xc_2b, yc_2b = center_2b
    Ri_2b        = calc_R(*center_2b)
    R_2b         = Ri_2b.mean()
    residu_2b    = sum((Ri_2b - R_2b)**2)

    if (R_2b < 1.0) or (R_2b > 5.0):
        raise ValueError('Bad circle fit')

    #- Convert center back into CS5 x,y
    xc, yc = uv2xy(xc_2b, yc_2b)
    # xc, yc = xc_2b, yc_2b

    return xc, yc, R_2b

def fast_fit_circle(x,y) :
    # init
    nn=len(x)
    i1=np.arange(nn)
    ### i2=(i1+1)%nn
    i2=(i1+nn//2-1)%nn
    
    # midpoints
    mx=((x[i1]+x[i2])/2.)
    my=((y[i1]+y[i2])/2.)
    nx=(y[i2]-y[i1])
    ny=-(x[i2]-x[i1])

    # solve for intersection of perpendicular bisectors
    # with s1,s2 are affine parameters of 2 adjacent perpendicular bisectors
    # 2 equations:
    # mx1 + nx1*s1 = mx2 + nx2*s2
    # my1 + ny1*s1 = my2 + ny2*s2
    s1 = (ny[i2]*mx[i2]-nx[i2]*my[i2]-ny[i2]*mx[i1]+nx[i2]*my[i1])/(ny[i2]*nx[i1]-nx[i2]*ny[i1])

    # coordinates of intersections are estimates of center of circle
    xc=mx[i1]+nx[i1]*s1[i1]
    yc=my[i1]+ny[i1]*s1[i1]
    
    # first estimate of center is mean of all intersections
    xc=np.mean(xc)
    yc=np.mean(yc)
    r=np.mean(np.sqrt((x-xc)**2+(y-yc)**2))

    #theta=np.linspace(0,2*np.pi,200)
    #plt.plot(xc,yc,"+",c="green")
    #plt.plot(r*np.cos(theta)+xc,r*np.sin(theta)+yc,c="green")

    # would need to refine this with chi2 minimization fit
   
    return xc,yc,r


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""FVC image processing""")
parser.add_argument('-i','--infile', type = str, default = None, required = True, nargs="*",
                    help = 'path to desimeter CSV files')
parser.add_argument('-o','--outfile', type = str, default = None, required = True,
                    help = 'path to output CSV ASCII file')
parser.add_argument('--plot', action = 'store_true', 
                    help = 'plot some circles')

args  = parser.parse_args()

x={}
y={}
xexp={}
yexp={}
first=True
for filename in args.infile :
    t=Table.read(filename)
    print(t.dtype.names)
    selection=(t["PINHOLE_ID"]==0)&(t["LOCATION"]>0)
    if first :
        for loc in t["LOCATION"][selection] :
            x[loc] = []
            y[loc] = []
            xexp[loc] = float(t["X_FP_EXP"][t["LOCATION"]==loc][0])
            yexp[loc] = float(t["Y_FP_EXP"][t["LOCATION"]==loc][0])
            print(loc,xexp[loc],yexp[loc])
        first=False

    for loc in t["LOCATION"][selection] :
        ii = np.where(t["LOCATION"]==loc)[0]
        if ii.size > 1 :
            print("several matched for LOCATION ",loc)
            continue
        i=ii[0]
        if not loc in x.keys() :
            x[loc] = []
            y[loc] = []
            xexp[loc] = float(t["X_FP_EXP"][t["LOCATION"]==loc][0])
            yexp[loc] = float(t["Y_FP_EXP"][t["LOCATION"]==loc][0])
            
        x[loc].append(float(t["X_FP"][i]))
        y[loc].append(float(t["Y_FP"][i]))

theta=np.linspace(0,2*np.pi,50)

locs=[]
x1=[]
y1=[]
x2=[]
y2=[]
for count,loc in enumerate(x.keys()) :
    if len(x[loc])<6 : continue
    x[loc]=np.array(x[loc])
    y[loc]=np.array(y[loc])
    # here is the fit
    try:
        xc,yc,r = fit_circle(x[loc],y[loc])
    except ValueError:
        continue
        
    if r<0.1 : continue
    print(loc,r)

    if args.plot and count < 40 :
        plt.figure("circles")
        plt.plot(x[loc],y[loc],"o")
        plt.plot(xexp[loc],yexp[loc],"x")
        theta=np.linspace(0,2*np.pi,50)
        plt.plot(xc+r*np.cos(theta),yc+r*np.sin(theta),"-",color="green")
        plt.plot(xc,yc,"+",color="green")
        
    
    x1.append(xexp[loc])
    y1.append(yexp[loc])
    x2.append(xc)
    y2.append(yc)
    locs.append(loc)
    
    if count>40 and args.plot : break

locs=np.array(locs)
x1=np.array(x1)
y1=np.array(y1)
x2=np.array(x2)
y2=np.array(y2)
dx=x2-x1
dy=y2-y1
dr=np.sqrt(dx**2+dy**2)
print("median offset=",np.median(dr))
ii=np.where(dr<0.2)[0]

# make a table out of that
t2=Table([locs[ii],x1[ii],y1[ii],x2[ii],y2[ii]],names=["LOCATION","X_FP_METRO","Y_FP_METRO","X_FP_THETA_AXIS","Y_FP_THETA_AXIS"],dtype=[int,float,float,float,float])

t2.write(args.outfile,format="csv",overwrite=True)
print("wrote",args.outfile)         

plt.figure(figsize=(6.1, 6))
plt.quiver(x1[ii],y1[ii],dx[ii],dy[ii])
plt.show()

    

