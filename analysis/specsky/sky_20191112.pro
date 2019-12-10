pro _john_orange_points, x, y
  x = [45.14519849325448, 40.23826381321665, 35.19561694148362, $
       30.084339581938966, 25.02339770005586, 20.11065508028785, $
       15.095983118255177, 11.09092467931874, 10.014326251334438]

  y = [-2.220446049250313e-16, -0.08645533141210393, -0.2132564841498561, $
       -0.3832853025936602, -1.0547550432276658, -1.3141210374639771, $
       -1.60806916426513, -1.8443804034582136, -1.8962536023054755]
end

pro sky_20191112

  str = mrdfits('/project/projectdirs/desi/users/ameisner/GFA/files/skymags-prelim.fits', 1)
  str = str[where((str.expid GE 27337) AND (str.expid LE 27396))]

  _john_orange_points, x, y

  plot, x, y, psym=3, charsize=2, xrange=[47.5, 7.5], /xst, $
        xtitle='target-Moon separation (deg)', $
        ytitle='relative sky brightness (mag)', yrange=[-2.2, 0.1], /yst, $
        title='20191112 ; 27337 ' + textoidl('\leq') + ' EXPID' + $
        textoidl('\leq') + ' 27396'
  oplot, x, y, psym=1, color=djs_icolor('orange')
  
  ; would be better to use per-GFA center RA, Dec rather than SKYRA, SKYDEC
  d = djs_diff_angle(str.ra_moon_deg, str.dec_moon_deg, str.skyra, str.skydec)

  offs = median(str[where(d GT 42)].skymag_median_top_camera)

  w2 = where(str.extname EQ 'GUIDE2')
  w5 = where(str.extname EQ 'GUIDE5')
  
  oplot, d[w5], str[w5].skymag_median_top_camera - offs, psym=1, color=djs_icolor('blue')
  oplot, d[w2], str[w2].skymag_median_top_camera - offs, psym=1, color=djs_icolor('magenta')

  xyouts, 45, -1.5, 'spectra (from John)', color=djs_icolor('orange'), $
          charsize=2
  
  xyouts, 45, -1.7, 'GUIDE2', color=djs_icolor('magenta'), $
          charsize=2

  xyouts, 45, -1.9, 'GUIDE5', color=djs_icolor('blue'), charsize=2

  write_png, 'sky_20191112.png', tvrd(true=1)
  
end
