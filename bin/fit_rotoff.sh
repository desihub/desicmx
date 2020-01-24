#!/bin/bash
#
# Help.
#
function usage() {
    local execName=$(basename $0)
    (
        echo "${execName} [-c] [-h] TILE NIGHT FVC GFA"
        echo ""
        echo "Wrapper script for running fit_rotoff.py."
        echo "Currently this script is intended for KPNO, not NERSC."
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
#
[[ -z "${DESI_SPECTRO_DATA}" ]] && source ${HOME}/setup.sh
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
if [[ ${#1} < 6 ]]; then
    tile=$(printf %06d ${1})
elif [[ ${#1} > 6 ]]; then
    echo "ERROR: Invalid tile '${1}'!" >&2
    exit 1
else
    tile=${1}
fi
if [[ ${#2} != 8 ]]; then
    echo "ERROR: Invalid night '${2}'!" >&2
    exit 1
else
    night=${2}
fi
if [[ ${#3} < 8 ]]; then
    fvc=$(printf %08d ${3})
elif [[ ${#3} > 8 ]]; then
    echo "ERROR: Invalid fvc exposure '${3}'!" >&2
    exit 1
else
    fvc=${3}
fi
if [[ ${#4} < 8 ]]; then
    gfa=$(printf %08d ${4})
elif [[ ${#4} > 8 ]]; then
    echo "ERROR: Invalid gfa exposure '${4}'!" >&2
    exit 1
else
    gfa=${4}
fi
#
# File names.
#
fvc_data=${DESI_SPECTRO_DATA}/${night}/${fvc}/fvc-${fvc}.fits.fz
csv=./fvc-${fvc}.csv
coordinates=${DESI_SPECTRO_DATA}/${night}/${fvc}/coordinates-${fvc}.fits
gfa_reduced=./realtime/${night}/${gfa}/gfa-${gfa}_reduced.fits
guide_reduced=./realtime/${night}/${gfa}/guide-${gfa}_reduced-00001.fits
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
[[ -f ${csv} ]] || desi_fvc_proc -i ${fvc_data} --exp ${coordinates} -o ${csv} --extname last
if [[ $? != 0 ]]; then
    echo "ERROR: desi_fvc_proc did not complete successfully, skipping fit_rotoff.py." >&2
    exit 1
fi
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
    -f ${fiberassign} \
    -a -50
