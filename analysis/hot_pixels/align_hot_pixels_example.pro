; instructions to blink aligned lab and post-installation data for
; this example case:
;
; IDL> align_hot_pixels_example, master_dark_cutout, ptc_cutout
; IDL> atv,master_dark_cutout, min=0,max=200,/al
; IDL> atv,ptc_cutout,/al,min=1000,max=6000

pro align_hot_pixels_example, master_dark_cutout, ptc_cutout

; this example intentionally hardcodes everything for PETAL_LOC = 3 <=> GFA02 (in Spain lab data nomenclature),
; specifically amplifier E

; single 14 second dark frame from GFA02 PTC lab sequence
; (not ideal, some apparently hot pixels could be CRs ...)

  fname_ptc = '/project/projectdirs/desi/engineering/gfa/PreProductionCamera/Data/99999999_GFA_FINAL/GFA02/PTC_031.fits'

  fname_master_dark = '/project/projectdirs/desi/users/ameisner/GFA/calib_20191015/GFA_master_dark.fits'

; ex = 3 of my master dark file is GUIDE3 (PETAL_LOC = 3)
  master_dark = readfits(fname_master_dark, ex=3)

; ex = 1 of Spain lab data is amplifier E
  ptc = readfits(fname_ptc, ex=1)

  master_dark_cutout = master_dark[50:1073, 0:515]
  ptc_cutout = ptc[50:1073, (508+8):1023] 

; note: no flips/transposes/rotations needed in this case...
  
end
