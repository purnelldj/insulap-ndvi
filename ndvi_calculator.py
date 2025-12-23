
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.warp import transform_bounds


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class NdviOutputs:
	geotiff_path: str
	png_preview_path: str


def crop_and_calculate_ndvi(
	input_raster: Union[str, Path],
	bbox: BBox,
	*,
	red_band: int,
	nir_band: int,
	output_geotiff: Union[str, Path],
	output_png: Union[str, Path],
	bbox_crs: Optional[Union[str, CRS]] = None,
	nodata_value: float = -9999.0,
) -> NdviOutputs:
	"""Crop a raster to a bounding box and compute NDVI.

	NDVI is computed as: (NIR - RED) / (NIR + RED).

	Args:
		input_raster: Path to a raster file (JP2, GeoTIFF, COG, etc.).
		bbox: (minx, miny, maxx, maxy) bounding box.
			  By default it's assumed to be in the raster's CRS.
		red_band: 1-based band index for the RED band.
		nir_band: 1-based band index for the NIR band.
		output_geotiff: Output path for NDVI GeoTIFF (float32).
		output_png: Output path for PNG preview (RGBA, 8-bit).
		bbox_crs: CRS of bbox (e.g. "EPSG:4326"). If provided and differs
			from the raster CRS, the bbox is reprojected.
		nodata_value: Nodata value written to NDVI GeoTIFF.

	Returns:
		NdviOutputs: paths to the GeoTIFF and PNG.
	"""

	input_raster = Path(input_raster)
	output_geotiff = Path(output_geotiff)
	output_png = Path(output_png)
	output_geotiff.parent.mkdir(parents=True, exist_ok=True)
	output_png.parent.mkdir(parents=True, exist_ok=True)

	if red_band < 1 or nir_band < 1:
		raise ValueError("Band indexes are 1-based and must be >= 1")

	minx, miny, maxx, maxy = bbox
	if not (minx < maxx and miny < maxy):
		raise ValueError(f"Invalid bbox {bbox}; expected (minx<maxx, miny<maxy)")

	with rasterio.open(input_raster) as src:
		if src.crs is None:
			raise ValueError("Input raster has no CRS; cannot crop by bounds")

		bounds_in_src_crs = (minx, miny, maxx, maxy)
		if bbox_crs is not None:
			bbox_crs_obj = CRS.from_user_input(bbox_crs)
			if bbox_crs_obj != src.crs:
				bounds_in_src_crs = transform_bounds(
					bbox_crs_obj,
					src.crs,
					minx,
					miny,
					maxx,
					maxy,
					densify_pts=21,
				)

		window = rasterio.windows.from_bounds(*bounds_in_src_crs, transform=src.transform)
		window = window.round_offsets().round_lengths()
		dataset_window = Window(col_off=0, row_off=0, width=src.width, height=src.height)
		window = window.intersection(dataset_window)

		if window.width <= 0 or window.height <= 0:
			raise ValueError("Crop window is empty; bbox may be outside raster extent")

		bands = src.read(
			indexes=[red_band, nir_band],
			window=window,
			masked=True,
			out_dtype="float32",
		)
		red = bands[0]
		nir = bands[1]

		with np.errstate(divide="ignore", invalid="ignore"):
			denom = nir + red
			ndvi = (nir - red) / denom

		# Mask pixels where denom is 0 or input was masked.
		invalid = np.ma.getmaskarray(red) | np.ma.getmaskarray(nir) | (denom == 0)
		ndvi = np.ma.array(ndvi, mask=invalid, dtype=np.float32)
		ndvi_filled = ndvi.filled(nodata_value).astype(np.float32)

		out_transform = src.window_transform(window)
		profile = src.profile.copy()
		profile.update(
			driver="GTiff",
			dtype="float32",
			count=1,
			height=int(window.height),
			width=int(window.width),
			transform=out_transform,
			nodata=float(nodata_value),
			compress="deflate",
			predictor=3,
			tiled=True,
		)

		with rasterio.open(output_geotiff, "w", **profile) as dst:
			dst.write(ndvi_filled, 1)

	_write_png_preview_rgba(
		ndvi_filled,
		nodata_value=float(nodata_value),
		output_path=output_png,
	)

	return NdviOutputs(str(output_geotiff), str(output_png))


def _write_png_preview_rgba(ndvi: np.ndarray, *, nodata_value: float, output_path: Path) -> None:
	"""Write an 8-bit RGBA preview from an NDVI array.

	- NDVI values are expected in [-1, 1] (values outside are clipped).
	- nodata_value becomes transparent.
	"""

	ndvi = np.asarray(ndvi)
	mask = ndvi == nodata_value

	scaled = np.clip((ndvi + 1.0) * 0.5, 0.0, 1.0)
	gray = (scaled * 255.0).astype(np.uint8)
	alpha = np.where(mask, 0, 255).astype(np.uint8)

	rgba = np.dstack([gray, gray, gray, alpha])
	Image.fromarray(rgba, mode="RGBA").save(output_path)


def _parse_bbox(bbox_str: str) -> BBox:
	parts = [p.strip() for p in bbox_str.split(",")]
	if len(parts) != 4:
		raise ValueError("bbox must be 'minx,miny,maxx,maxy'")
	return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))


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

	crop_and_calculate_ndvi(
		args.input,
		_parse_bbox(args.bbox),
		red_band=args.red_band,
		nir_band=args.nir_band,
		output_geotiff=args.out_tif,
		output_png=args.out_png,
		bbox_crs=args.bbox_crs,
		nodata_value=args.nodata,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

