from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from utils import (
    calculate_ndvi,
    crop_raster_to_bbox,
    find_band_paths_in_safe,
    parse_bbox,
    write_png_preview_rgba,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Compute NDVI from a Sentinel-2 .SAFE and write a PNG preview.")
    parser.add_argument("input", help="Input .SAFE directory path")
    parser.add_argument("--bbox", required=True, help="minx,miny,maxx,maxy")
    parser.add_argument("--out-png", required=True, help="Output PNG preview path")
    parser.add_argument("--bbox-crs", default=None, help="CRS of bbox, e.g. EPSG:4326. Default: raster CRS")
    parser.add_argument("--nodata", type=float, default=-9999.0, help="Nodata value to write into NDVI GeoTIFF")
    args = parser.parse_args(argv)

    red_path, nir_path = find_band_paths_in_safe(args.input)

    red_crop = crop_raster_to_bbox(
        red_path,
        parse_bbox(args.bbox),
        band_indexes=[1],
        bbox_crs=args.bbox_crs,
    )

    nir_crop = crop_raster_to_bbox(
        nir_path,
        parse_bbox(args.bbox),
        band_indexes=[1],
        bbox_crs=args.bbox_crs,
    )

    if red_crop.bands.shape != nir_crop.bands.shape:
        raise ValueError(
            "Cropped red and NIR bands have different shapes; "
            "bbox/resolution may be mismatched between B04 and B08."
        )
    if red_crop.transform != nir_crop.transform:
        raise ValueError(
            "Cropped red and NIR bands have different geotransforms; "
            "bbox/resolution may be mismatched between B04 and B08."
        )

    ndvi = calculate_ndvi(red_crop.bands[0], nir_crop.bands[0], nodata_value=args.nodata)
    _ = write_png_preview_rgba(ndvi, nodata_value=args.nodata, output_path=args.out_png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

