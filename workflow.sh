#!/usr/bin/env bash

set -x -e

WORKFLOW=$(dirname $(readlink -f "$0"))
WORKER_DIR="/home/worker"
IN_DIR="${WORKER_DIR}/workDir/inDir"
OUT_DIR="${WORKER_DIR}/workDir/outDir"
WPS_PROPS="${WORKER_DIR}/workDir/FSTEP-WPS-INPUT.properties"
PROC_DIR="${WORKER_DIR}/procDir"
TIMESTAMP=$(date --utc +%Y%m%d_%H%M%SZ)

# Input parameters available as shell variables
source ${WPS_PROPS}

# below is all stuff that was in the example but doesn't make sense to me
# mkdir -p ${IN_DIR}/input
# mkdir -p ${OUT_DIR}/output
# cat ${WPS_PROPS}
# ls -l ${IN_DIR}/input/
# cp -r ${IN_DIR}/input/* ${OUT_DIR}/output
# finish

# input patams: s2prod, BBOX
# BBOX read from WPS_PROPS
# outputs: ndvipng

# s2 product will in theory be automatically saved to 
S2PRODPATH=${IN_DIR}/s2prod

# output png should be saved at
OUTPNGPATH=${OUT_DIR}/ndvipng/ndvi.png

# run the script with the correct paths
python3 ndvi_calculator.py ${S2PRODPATH} \
    --bbox ${BBOX} \
    --out-png ${OUTPNGPATH}
