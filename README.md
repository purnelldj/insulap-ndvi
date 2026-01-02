# insulap-ndvi

Compute NDVI from a Sentinel-2 `.SAFE` product, cropped to a bounding box, and write:
- PNG preview (8-bit grayscale RGBA; nodata transparent)

## Option A: Docker (GDAL base image)

Build:

```bash
docker build -t insulap-ndvi .
```

### Run NDVI inside the container (mount a local directory)

This example mounts a local folder `./data/workDir` into the container at `/home/worker/workDir`, finds the first Sentinel-2 `.SAFE` folder inside it, and writes outputs back into the same folder.

```bash
mkdir -p ./.data/workDir/inDir/s2prod
scp ./FSP-WPS-INPUT.properties ./.data/workDir

# Put a SAFE folder at: ./.data/workDir/inDir/s2prod/S2A.....SAFE
# e.g., S2A_MSIL2A_20240728T105031_N0511_R051_T31UDQ_20240728T183146.SAFE

docker run --rm \
  -v "$PWD/.data/workDir:/home/worker/workDir" \
  insulap-ndvi
```
This crops over a 5km x 5km region in Paris, calculates NDVI and then saves the resuls to a png file.

If your bbox is in WGS84 (lon/lat), pass `--bbox-crs EPSG:4326`.

## Option B: Micromamba (environment.yml)

Create the environment:

```bash
micromamba create -f environment.yml
```

Activate:

```bash
micromamba activate insulap-ndvi
```

Run locally:

```bash
python ndvi_calculator.py ./.data \
  --bbox "500000,4100000,510000,4110000" \
  --out-png ndvi.png
```

## Notes

- Bands are auto-detected from the `.SAFE` layout by filename:
  - Red: `*B04*.jp2`
  - NIR: `*B08*.jp2`
- NDVI uses `(nir - red) / (nir + red)`.
- Output nodata defaults to `-9999.0` (set via `--nodata`).
