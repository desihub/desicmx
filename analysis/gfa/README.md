# GFA Analysis

## Calibration

This directory contains the following notebooks for GFA calibration analysis:
 - GFA-Zero-Calibration: analyze zeros to calculate master bias residual images and estimate readnoise per amplifier.
 - GFA-Flat-Calibration: analyze dome flats with increasing exposure times to estimate gain per amplifier.
 - GFA-Dark-Calibration: analyze 5s darks to estimate dark current temperature dependence parameters and calculate master dark images.
 - GFA-Pixel-Mask: analyze dome darks to identify bad pixels and study their spatial and temporal structure.
These notebooks should be run in the order above to reproduce the existing calibrations or to incorporate new data.
 
All results are saved in a single FITS file. For details, see DESI-5315.  Notebooks use code from https://github.com/dkirkby/desietcimg.

## ETC Pipeline

The `GFA-ETC-Pipeline.ipynb` notebook has examples of using the ETC online pipeline to apply these calibrations.

## Guide Frame Centroids

The `Guider-Performance-Studies.ipynb` notebook has examples of comparing the centroids calculated by different codes on long guide sequences.
