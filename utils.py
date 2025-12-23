
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.warp import transform_bounds


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class CropResult:
	bands: np.ma.MaskedArray
	transform: rasterio.Affine
	crs: CRS
	src_profile: dict


def parse_bbox(bbox_str: str) -> BBox:
	parts = [p.strip() for p in bbox_str.split(",")]
	if len(parts) != 4:
		raise ValueError("bbox must be 'minx,miny,maxx,maxy'")
	bbox = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
	minx, miny, maxx, maxy = bbox
	if not (minx < maxx and miny < maxy):
		raise ValueError(f"Invalid bbox {bbox}; expected (minx<maxx, miny<maxy)")
	return bbox


def crop_raster_to_bbox(
	input_raster: Union[str, Path],
	bbox: BBox,
	*,
	band_indexes: Sequence[int],
	bbox_crs: Optional[Union[str, CRS]] = None,
	out_dtype: str = "float32",
) -> CropResult:
	"""Read a crop window for the given bbox and return requested bands.

	Args:
		input_raster: Path to raster.
		bbox: (minx, miny, maxx, maxy) in raster CRS unless bbox_crs is set.
		band_indexes: 1-based band indices to read.
		bbox_crs: CRS of bbox (e.g. "EPSG:4326"). If provided and differs from
			raster CRS, bbox is reprojected into raster CRS before windowing.
		out_dtype: dtype used when reading bands.
	"""

	if len(band_indexes) == 0:
		raise ValueError("band_indexes must be non-empty")
	if any(b < 1 for b in band_indexes):
		raise ValueError("Band indexes are 1-based and must be >= 1")

	input_raster = Path(input_raster)
	minx, miny, maxx, maxy = bbox

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
			indexes=list(band_indexes),
			window=window,
			masked=True,
			out_dtype=out_dtype,
		)
		transform = src.window_transform(window)

		return CropResult(
			bands=bands,
			transform=transform,
			crs=src.crs,
			src_profile=src.profile.copy(),
		)


def calculate_ndvi(
	red: np.ma.MaskedArray,
	nir: np.ma.MaskedArray,
	*,
	nodata_value: float = -9999.0,
) -> np.ndarray:
	"""Compute NDVI as (nir-red)/(nir+red), filling invalid pixels with nodata."""

	red = np.ma.array(red, copy=False)
	nir = np.ma.array(nir, copy=False)

	with np.errstate(divide="ignore", invalid="ignore"):
		denom = nir + red
		ndvi = (nir - red) / denom

	denom_is_zero = np.ma.array(denom == 0).filled(False)
	invalid = np.ma.getmaskarray(red) | np.ma.getmaskarray(nir) | denom_is_zero
	ndvi = np.ma.array(ndvi, mask=invalid, dtype=np.float32)
	return ndvi.filled(nodata_value).astype(np.float32)


def build_ndvi_geotiff_profile(
	src_profile: dict,
	*,
	height: int,
	width: int,
	transform: rasterio.Affine,
	nodata_value: float,
) -> dict:
	profile = src_profile.copy()
	profile.update(
		driver="GTiff",
		dtype="float32",
		count=1,
		height=int(height),
		width=int(width),
		transform=transform,
		nodata=float(nodata_value),
		compress="deflate",
		predictor=3,
		tiled=True,
	)
	return profile


def write_geotiff(output_path: Union[str, Path], array: np.ndarray, *, profile: dict) -> str:
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with rasterio.open(output_path, "w", **profile) as dst:
		dst.write(np.asarray(array, dtype=np.float32), 1)
	return str(output_path)


def write_png_preview_rgba(
	ndvi: np.ndarray,
	*,
	nodata_value: float,
	output_path: Union[str, Path],
) -> str:
	"""Write an 8-bit RGBA preview from an NDVI array.

	- NDVI values are expected in [-1, 1] (values outside are clipped).
	- nodata_value becomes transparent.
	"""

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	ndvi = np.asarray(ndvi)
	mask = ndvi == nodata_value

	scaled = np.clip((ndvi + 1.0) * 0.5, 0.0, 1.0)
	gray = (scaled * 255.0).astype(np.uint8)
	alpha = np.where(mask, 0, 255).astype(np.uint8)

	rgba = np.dstack([gray, gray, gray, alpha])
	Image.fromarray(rgba, mode="RGBA").save(output_path)
	return str(output_path)

