# insulap-ndvi

Compute NDVI from a raster (JP2/COG/GeoTIFF), cropped to a bounding box, and write:
- NDVI GeoTIFF (float32)
- PNG preview (8-bit grayscale RGBA; nodata transparent)

## Option A: Docker (GDAL base image)

Build:

```bash
docker build -t insulap-ndvi .
```

Sanity check (imports only):

```bash
docker run --rm insulap-ndvi
```

### Run NDVI inside the container (mount a local directory)

This example mounts a local folder `./data` into the container at `/data`, reads an input raster from it, and writes outputs back into the same folder.

```bash
mkdir -p data

# Put an input raster at: ./data/input.tif (or .jp2 / COG)

docker run --rm \
  -v "$PWD/data:/data" \
  insulap-ndvi \
  python3.11 ndvi_calculator.py /data/input.tif \
    --bbox "500000,4100000,510000,4110000" \
    --red-band 3 \
    --nir-band 4 \
    --out-tif /data/ndvi.tif \
    --out-png /data/ndvi.png
```

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
python ndvi_calculator.py input.tif \
  --bbox "500000,4100000,510000,4110000" \
  --red-band 3 \
  --nir-band 4 \
  --out-tif ndvi.tif \
  --out-png ndvi.png
```

## Notes

- Band indexes are **1-based**.
- NDVI uses `(nir - red) / (nir + red)`.
- Output nodata defaults to `-9999.0` (set via `--nodata`).
