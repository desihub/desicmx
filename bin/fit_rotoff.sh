#!/bin/bash
#
# Help.
#
function usage() {
    local execName=$(basename $0)
    (
        echo "${execName} [-c] [-h] TILE NIGHT FVC GFA"
        echo ""
        echo "Run fit_rotoff.py."
        echo ""
        echo "    -c = GFA exposure is a data cube (guide file)."
        echo "    -h = Print this message and exit."
        echo "  TILE = Fiberassign tile number."
        echo " NIGHT = Night of observation."
        echo "   FVC = FVC exposure number."
        echo "   GFA = GFA exposure number."
    ) >&2
}
#
# Load an approximation of the NERSC desiconda stack.
# The stack at NERSC is probably quite old.
#
[[ -z "${DESI_SPECTRO_DATA}" ]] && source ./setup.sh
#
# Load desimeter and desicmx.
#
if [[ -z "${DESICMX}" ]]; then
    module use ${HOME}/users/bweaver/modulefiles
    module load desimeter desicmx
fi
#
# Options.
#
cube=/bin/false
while getopts ch argname; do
    case ${argname} in
        c) cube=/bin/true ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND-1))
#
# Command-line arguments.
#
n_tile=$(echo -n $1 | wc -m)
if [[ ${n_tile} < 6 ]]; then
    tile=$(printf %06d $1)
elif [[ ${n_tile} > 6 ]]; then
    echo "ERROR: Invalid tile '$1'!"
    exit 1
else
    tile=$1
fi
n_night=$(echo -n $2 | wc -m)
if [[ ${n_night} != 8 ]]; then
    echo "ERROR: Invalid night '$2'!"
    exit 1
else
    night=$2
fi
n_fvc=$(echo -n $3 | wc -m)
if [[ ${n_fvc} < 8 ]]; then
    fvc=$(printf %08d $3)
elif [[ ${n_fvc} > 8 ]]; then
    echo "ERROR: Invalid fvc exposure '$3'!"
    exit 1
else
    fvc=$3
fi
n_gfa=$(echo -n $4 | wc -m)
if [[ ${n_gfa} < 8 ]]; then
    gfa=$(printf %08d $4)
elif [[ ${n_gfa} > 8 ]]; then
    echo "ERROR: Invalid gfa exposure '$4'!"
    exit 1
else
    gfa=$4
fi
#
# File names.
#
fvc_data=${DESI_SPECTRO_DATA}/${night}/${fvc}/fvc-${fvc}.fits.fz
csv=./fvc-${fvc}.csv
coordinates=${DESI_SPECTRO_DATA}/${night}/${fvc}/coordinates-${fvc}.fits
gfa_reduced=./realtime/${night}/${gfa}/gfa-${gfa}_reduced.fits
guide_reduced=./realtime/${night}/${gfa}/guide-${gfa}_reduced-00000.fits
if [[ ${tile:0:4} == "0635" ]]; then
    fiberassign=./code/tiles/063/0635/fiberassign-${tile}.fits
else
    fiberassign=./code/tiles/063/fiberassign-${tile}.fits
fi
if ${cube}; then
    reduced=${guide_reduced}
else
    reduced=${gfa_reduced}
fi
#
# FVC processing.
#
while [[ ! -f ${fvc_data} ]]; do
    echo "Waiting for fvc data: ${fvc_data} ..."
    sleep 5
done
[[ -f ${csv} ]] || desi_fvc_proc -i ${fvc_data} --exp ${coordinates} -o ${csv}
#
# Main command.
#
while [[ ! -f ${reduced} ]]; do
    echo "Waiting for gfa reduction: ${reduced} ..."
    sleep 5
done
python ${DESICMX}/analysis/dither/fit_rotoff.py \
    -d ${csv} \
    -c ${coordinates} \
    -g ${reduced} \
    -f ${fiberassign}
