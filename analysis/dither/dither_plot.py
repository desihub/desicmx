import csv,sys,os
import argparse
from math import sqrt
from pprint import pprint
import numpy as np
import fitsio
import matplotlib.pyplot as plt
from pandas import DataFrame
import pylab as py
from matplotlib import gridspec

   

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
            py.savefig(figoutdir+output_filename)
         else:                  
            print('None of the fibers in '+'qa-000'+str(expid)+'.fits'+' have landed on a source (snr >0.5) in petal '+petalnum)
            continue
         plt.show()
                  




def plot_dither_seq(exposure_sequence, obsday, petalnum_list, channel, tileid, nightwatchdir,fiberassign_dir, plotout_dir,snr_thresh):
        
        
    

     '''NOTE: this hack to replace the csv file with fiberassign's raw output 
              and it only works if the originally assigned tile is also supplied.
              For all the dithered tiles that are provided (see ), 
              the originally assigned tile is supplied and its tile ID is one 
              behind the dithered Tile ID (e.g., original = 63069, dithered tile = 63070)
              
              
              Example run:  
              
                   tileid, obsday, petalnum_list, channel = 63068, '20191126', [0,2,7,9], b'B' 
              
                   nightwatchdir = '/exposures/nightwatch/'
              
                   fiberassign_dir = '/data/tiles/ALL_tiles/20191119/'
                    
                   plotoutdir = './'

                   exposure_sequence = [30180,30156,30148,30140,30136,30132] 

                   plot_dither_seq(exposure_sequence, obsday, petalnum_list, channel, tileid, nightwatchdir, fiberassign_dir, plotout_dir, snr_thresh= 0.5)'''

        
        
    print('original assignment is: '+fiberassign_dir+"/fiberassign-0{}.fits".format(str(tileid-1)))
    print('Dithered assignment is: '+fiberassign_dir+"/fiberassign-0{}.fits".format(str(tileid)))
    print('if these are not the correct files to be used, stop now!')
                
   
    for expid in exposure_sequence:
        
        fig = py.figure(figsize=(10, 25), dpi=100)    
        fig.subplots_adjust(wspace=0.3,hspace=0.3, top=0.97, bottom=0.07, left=0.08, right=0.98)
        gs = gridspec.GridSpec(5,2) 
        ip = 0
        
        
        for petalnum in petalnum_list:

            petal_assign = {}  
            if ( not os.path.exists(fiberassign_dir+"/fiberassign-0{}.fits".format(str(tileid)))) | (not os.path.exists(fiberassign_dir+"/fiberassign-0{}.fits".format(str(tileid-1)))):
                
                raise ValueError('original assignment {} or dithered assignment {} do not exist!'.format(fiberassign_dir+"/fiberassign-0{}.fits".format(str(tileid-1)),fiberassign_dir+"/Fiberassign-0{}.fits".format(str(tileid))))
            
            else:
                dit= fitsio.read(fiberassign_dir+"/fiberassign-0{}.fits".format(str(tileid)),ext=1)
                orig = fitsio.read(fiberassign_dir+"/fiberassign-0{}.fits".format(str(tileid-1)), ext=1)
                p = (dit['PETAL_LOC'] == petalnum)

                if np.sum(p) != 500:
                    raise ValueError('Something is wrong with {} '.format(fiberassign_dir+"/Fiberassign-0{}.fits".format(str(tileid))))


                for i in dit[p]['FIBER']:    

                    fiber = i
                    x = orig['FIBERASSIGN_X'][i]
                    y = orig['FIBERASSIGN_Y'][i]
                    dra  = (dit['TARGET_RA'][i]-orig['TARGET_RA'][i])*np.cos(orig['TARGET_DEC'][i]*np.pi/180)*3600
                    ddec =(dit['TARGET_DEC'][i]-orig['TARGET_DEC'][i])*3600
                    if dra*dra+ddec*ddec<225:
                        petal_assign[fiber] = {}
                        petal_assign[fiber]['r'] = sqrt(x*x+y*y)
                        petal_assign[fiber]['inner'] = petal_assign[fiber]['r']<285
                        petal_assign[fiber]['dra'] = dra             
                        petal_assign[fiber]['ddec'] = ddec


                nightwatch_filename  =   nightwatchdir + obsday + '/000'+str(expid)+'/qa-000'+str(expid)+'.fits'
                output_filename = 'plot'+str(expid)+'.png'

                nw = fitsio.FITS(nightwatch_filename)
                data = nw['PER_CAMFIBER'].read()
                fibers = data['FIBER']
                snrs = data['MEDIAN_CALIB_SNR']
                cams = data['CAM']
                for fiber, snr, cam in zip(fibers, snrs, cams):
                    if cam == channel and fiber in petal_assign:
                        camera=channel
                        petal_assign[fiber]['snr'] = snr

                df = DataFrame.from_dict(petal_assign, orient='index')
                ax = plt.subplot(gs[ip]); ip += 1
                plot_title = obsday+' - Exp '+str(expid)+'- PETAL_LOC '+str(petalnum)+' Cam: '+str(camera.decode())
                ax.set_title(plot_title, fontsize=15)

                if ((df[df['snr']>snr_thresh].size >0) & (df[df['snr']>snr_thresh][df['r']<225].size >0)) :
                    plot_title = obsday+' - Exp '+str(expid)+'- PETAL_LOC '+str(petalnum)+' Cam: '+str(camera.decode())
                    ii = (np.abs(df['dra']) > 3.5* np.std(df['dra'])) |  (np.abs(df['ddec']) > 3.5* np.std(df['ddec']))
                    df = df[~ii]

                    
                    df.plot(kind='scatter', x='dra', y='ddec', s=3, color='gray', ax=ax)
                    df[df['snr']>snr_thresh].plot(kind='scatter', x='dra', y='ddec', color='blue', ax=ax)
                    df[df['snr']>snr_thresh][df['r']<225].plot(kind='scatter', x='dra', y='ddec', color='red', ax=ax)

                    xlim = (np.min(df['dra'])-0.5,np.max(df['dra'])+0.5 )
                    ylim = (np.min(df['ddec'])-0.5,np.max(df['ddec'])+0.5 )

                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    
                    ax.set_xlabel('delta_RA [arcsec]',fontsize=18)
                    ax.set_ylabel('delta_Dec [arcsec]',fontsize=18)
                    ax.minorticks_on()
                    ax.tick_params(which='major', length=8, width=1.5, direction='in') 
                    ax.tick_params(which='minor', length=6, color='#000033', width=1.0, direction='in')        
                    x_ax = ax.twiny()
                    x_ax.minorticks_on()
                    x_ax.tick_params(which='major', length=8, width=1.0, direction='in')
                    x_ax.tick_params(which='minor', length=6, color='#000033', width=1.0, direction='in')     
                    x_ax.tick_params()
                    y_ax= ax.twinx()
                    y_ax.minorticks_on() 
                    x_ax.set_xlim(xlim)
                    y_ax.set_ylim(ylim)
                    y_ax.tick_params(which='major', length=8, width=1.0, direction='in')
                    y_ax.tick_params(which='minor', length=6, color='#000033', width=1.0, direction='in')
                    
                    for tick in ax.xaxis.get_major_ticks():
                         tick.label.set_fontsize(16) 
                            
                    for tick in ax.yaxis.get_major_ticks():
                         tick.label.set_fontsize(16) 
                            
                    for tick in x_ax.xaxis.get_major_ticks():
                         tick.label.set_fontsize(16) 
    
                    for tick in y_ax.yaxis.get_major_ticks():
                         tick.label.set_fontsize(16) 
    
            
        
                else:                  
                    print('None of the fibers in '+'qa-000'+str(expid)+'.fits'+' have landed on a source (snr >0.5) in petal '+str(petalnum))

                    continue
                    
        pylab.savefig(plotout_dir+output_filename)

        plt.show()
        print('_________________________________________________________________________________')  
        print('')








