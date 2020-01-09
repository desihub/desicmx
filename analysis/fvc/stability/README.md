## Processing FVC stability data from 20200107

Stephen Bailey

```
basedir=/project/projectdirs/desi/spectro/data/20200107
cd /project/projectdirs/desi/cmx/fvc/stability/20200107
for e in $(seq 38674 38874); do
    expid=000$e
    outfile=spots-$expid.csv
    if [ -f $outfile ]; then
        echo Skipping $outfile
    else
        desi_fvc_proc \
            --infile $basedir/$expid/fvc-$expid.fits.fz \
            --expected $basedir/$expid/coordinates-$expid.fits \
            --outfile $outfile
    fi
done
```

## SCRATCH

```
basedir=/project/projectdirs/desi/spectro/data/20200107
expid=00038674
desi_fvc_proc \
    --infile $basedir/$expid/fvc-$expid.fits.fz \
    --expected $basedir/$expid/coordinates-$expid.fits \
    --outfile spots-$expid.csv

x = Table.read('spots-00038674.csv')
x['XPIX', 'YPIX', 'PETAL_LOC', 'DEVICE_LOC', 'PINHOLE_ID'][x['LOCATION'] == 533]
```
