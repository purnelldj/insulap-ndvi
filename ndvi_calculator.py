
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from utils import (
    build_ndvi_geotiff_profile,
    calculate_ndvi,
    crop_raster_to_bbox,
    parse_bbox,
    write_geotiff,
    write_png_preview_rgba,
)


@dataclass(frozen=True)
class NdviOutputs:
    geotiff_path: str
    png_preview_path: str


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Crop raster and compute NDVI (GeoTIFF + PNG preview).")
    parser.add_argument("input", help="Input raster path (JP2/COG/GeoTIFF/etc)")
    parser.add_argument("--bbox", required=True, help="minx,miny,maxx,maxy")
    parser.add_argument("--red-band", type=int, required=True, help="1-based red band index")
    parser.add_argument("--nir-band", type=int, required=True, help="1-based NIR band index")
    parser.add_argument("--out-tif", required=True, help="Output NDVI GeoTIFF path")
    parser.add_argument("--out-png", required=True, help="Output PNG preview path")
    parser.add_argument("--bbox-crs", default=None, help="CRS of bbox, e.g. EPSG:4326. Default: raster CRS")
    parser.add_argument("--nodata", type=float, default=-9999.0, help="Nodata value to write into NDVI GeoTIFF")
    args = parser.parse_args(argv)

    crop = crop_raster_to_bbox(
        args.input,
        parse_bbox(args.bbox),
        band_indexes=[args.red_band, args.nir_band],
        bbox_crs=args.bbox_crs,
    )

    ndvi = calculate_ndvi(crop.bands[0], crop.bands[1], nodata_value=args.nodata)
    profile = build_ndvi_geotiff_profile(
        crop.src_profile,
        height=ndvi.shape[0],
        width=ndvi.shape[1],
        transform=crop.transform,
        nodata_value=args.nodata,
    )

    geotiff_path = write_geotiff(args.out_tif, ndvi, profile=profile)
    png_preview_path = write_png_preview_rgba(ndvi, nodata_value=args.nodata, output_path=args.out_png)
    _ = NdviOutputs(geotiff_path=geotiff_path, png_preview_path=png_preview_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

