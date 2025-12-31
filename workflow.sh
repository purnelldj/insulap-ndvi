#!/usr/bin/env bash

set -x -e

# two input params: s2 product and bbox

# s2 product will be automatically saved to 
S2PRODPATH=/home/worker/workDir/inDir/s2prod

# bbox will be stored in as BBOX
source /home/worker/workDir/FSTEP-WPS-INPUT.properties

# run the script with the correct paths
python3 ndvi_calculator.py ${S2PRODPATH}
    --bbox ${BBOX} \
    --out-png /data/ndvi.png
