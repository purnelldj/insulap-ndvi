#!/usr/bin/env bash

set -x -e

# input patams: s2prod, BBOX
# outputs: ndvipng

# s2 product will be automatically saved to 
S2PRODPATH=/home/worker/workDir/inDir/s2prod

# bbox will be stored in the following file as BBOX
source /home/worker/workDir/FSTEP-WPS-INPUT.properties

# output png should be saved at
OUTPNGPATH=/home/worker/workDir/outDir/ndvipng/ndvi.png

# run the script with the correct paths
python3 ndvi_calculator.py ${S2PRODPATH} \
    --bbox ${BBOX} \
    --out-png ${OUTPNGPATH}
