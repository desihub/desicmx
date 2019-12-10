import csv,sys,os
import argparse
from math import sqrt
from pprint import pprint
import numpy as np
import fitsio
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame




#seq =[29787, 29791, 29795, 29799, 29803, 29807, 29809, 29812, 29816, 29818, 29822] ### 20191124
#seq =[29736, 29742, 29744, 29748, 29752, 29756, 29760, 29764, 29770, 29774, 29778]  ### 20191124
#seqlist = [30180,30156,30148,30140,30136,30132]


#petal_assign = {}
#obsday,petalnum,channel = '20191126',"2",b'B' 

#path = '/exposures/nightwatch/'+obsday+'/' 
#home = '/n/home/desiobserver/sarahE/'+obsday+'/'    

def grokdither(seqlist, obsday, petalnum, channel, fiberassigncsvfilename, nightwatchroot='/exposures/nightwatch/', figoutdir='/n/home/desiobserver/dithertest/'):
   petal_assign = {}   
   with open(fiberassigncsvfilename) as dithers:
      
      reader = csv.DictReader(dithers, skipinitialspace=True)
      for line in reader:
         if line['PETAL'] == petalnum:
            
            fiber = int(line['FIBER'])
            x = float(line['OROGINALLY_ASSINED_FIBER_X'])
            y = float(line['ORIGINALLY_ASSIGNED_FIBER_Y'])
            dra  = float(line['DELTA_RA_DITHERED_ORIGINAL_degrees'])*3600
            ddec = float(line['DELTA_DEC_DITHERED_ORIGINAL_degrees'])*3600
            if dra*dra+ddec*ddec<225:
               petal_assign[fiber] = {}
               
               petal_assign[fiber]['r'] = sqrt(x*x+y*y)
               
               petal_assign[fiber]['inner'] = petal_assign[fiber]['r']<285
               
               petal_assign[fiber]['dra'] = dra
               
               petal_assign[fiber]['ddec'] = ddec
               
               
      for expid in seqlist:

         nightwatch_filename  =   nightwatchroot + obsday + '/000'+str(expid)+'/qa-000'+str(expid)+'.fits'
         output_filename = 'plot'+str(expid)+'.png'
         
         nw = fitsio.FITS(nightwatch_filename)
         data = nw['PER_CAMFIBER'].read()
         fibers = data['FIBER']
         snrs = data['MEDIAN_CALIB_SNR']
         cams = data['CAM']
         for fiber, snr, cam in zip(fibers, snrs, cams):
            #print(str(fiber)+' '+str(snr)+' '+str(cam))
            if cam == channel and fiber in petal_assign:
               camera=channel
               petal_assign[fiber]['snr'] = snr
               
         #print(foo)
         df = DataFrame.from_dict(petal_assign, orient='index')
         if ((df[df['snr']>0.5].size >0) & (df[df['snr']>0.5][df['r']<225].size >0)) :
            plot_title = obsday+' - Exp '+str(expid)+'- PETAL_LOC '+petalnum+' Cam: '+str(camera.decode())
            ii = (np.abs(df['dra']) > 3.5* np.std(df['dra'])) |  (np.abs(df['ddec']) > 3.5* np.std(df['ddec']))
            df = df[~ii]
            
            ax = df.plot(kind='scatter', x='dra', y='ddec', s=3, color='gray')
            df[df['snr']>0.5].plot(kind='scatter', x='dra', y='ddec', color='blue', ax=ax)
            df[df['snr']>.5][df['r']<225].plot(kind='scatter', x='dra', y='ddec', color='red', ax=ax)
            
            xlim = (np.min(df['dra'])-0.5,np.max(df['dra'])+0.5 )
            ylim = (np.min(df['ddec'])-0.5,np.max(df['ddec'])+0.5 )
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title(plot_title)
            ax.set_xlabel('delta ra, arcsec')
            ax.set_ylabel('delta dec, arcsec')
            ax.minorticks_on()
            ax.tick_params(which='major', length=8, width=1.5, direction='in') 
            ax.tick_params(which='minor', length=6, color='#000033', width=1.0, direction='in')        
            x_ax = ax.twiny()
            x_ax.minorticks_on()
            x_ax.tick_params(which='major', length=8, width=1.0, direction='in')
            x_ax.tick_params(which='minor', length=6, color='#000033', width=1.0, direction='in')     
            x_ax.tick_params()
            y_ax= plt.twinx()
            y_ax.minorticks_on() 
            x_ax.set_xlim(xlim)
            y_ax.set_ylim(ylim)
            y_ax.tick_params(which='major', length=8, width=1.0, direction='in')
            y_ax.tick_params(which='minor', length=6, color='#000033', width=1.0, direction='in')
            pylab.savefig(figoutdir+output_filename)
         else:                  
            print('None of the fibers in '+'qa-000'+str(expid)+'.fits'+' have landed on a source (snr >0.5) in petal '+petalnum)
            continue
         plt.show()
                  
