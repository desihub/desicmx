#!/usr/bin/env python
import csv
import argparse
from math import sqrt
from pprint import pprint

import numpy as np
import fitsio
import matplotlib.pyplot as plt
import pylab
from pandas import DataFrame


parser = argparse.ArgumentParser()
parser.add_argument('obsday', type=int, help='YYYYMMDD')
parser.add_argument('expid', type=int, help='Exposure number')
args = parser.parse_args()

nightwatch_filename = '/exposures/nightwatch/{}/{expid:08d}/qa-{expid:08d}.fits'.format(args.obsday, expid=args.expid)
output_filename = 'plot_{}_{expid:08d}.png'.format(args.obsday, expid=args.expid)
plot_title = 'Obsday {} Exposure {expid:08d}'.format(args.obsday, expid=args.expid)

petal0_assign = {}
with open('5n1arcsec_dither_assignment_diff_63068.csv') as dithers:
    reader = csv.DictReader(dithers, skipinitialspace=True)
    for line in reader:
        if line['PETAL'] == "0":
            fiber = int(line['FIBER'])
            x = float(line['OROGINALLY_ASSINED_FIBER_X'])
            y = float(line['ORIGINALLY_ASSIGNED_FIBER_Y'])
            dra  = float(line['DELTA_RA_DITHERED_ORIGINAL_degrees'])*3600
            ddec = float(line['DELTA_DEC_DITHERED_ORIGINAL_degrees'])*3600

            if dra*dra+ddec*ddec<225:
                petal0_assign[fiber] = {}
                petal0_assign[fiber]['r'] = sqrt(x*x+y*y)
                petal0_assign[fiber]['inner'] = petal0_assign[fiber]['r']<285
                petal0_assign[fiber]['dra'] = dra
                petal0_assign[fiber]['ddec'] = ddec
         

nw = fitsio.FITS(nightwatch_filename)
data = nw[2].read()
fibers = data['FIBER']
snrs = data['MEDIAN_CALIB_SNR']
cams = data['CAM']
for fiber, snr, cam in zip(fibers, snrs, cams):
    if cam == b'R' and fiber in petal0_assign:
        petal0_assign[fiber]['snr'] = snr

df = DataFrame.from_dict(petal0_assign, orient='index')
print(df)

ax = df.plot(kind='scatter', x='dra', y='ddec', s=3, color='black')
df[df['snr']>0.5].plot(kind='scatter', x='dra', y='ddec', color='blue', ax=ax)
df[df['snr']>0.5][df['r']<255].plot(kind='scatter', x='dra', y='ddec', color='red', ax=ax)

#plt.scatter(dra[(snr>0.5)&(r<225)], ddec[(snr>0.5)&(r<225)], color='red')
plt.title(plot_title)
plt.xlabel('delta ra')
plt.ylabel('delta dec')
pylab.savefig(output_filename)
#plt.show()
