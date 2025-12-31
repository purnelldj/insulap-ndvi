
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.warp import transform_bounds


BBox = Tuple[float, float, float, float]


def find_band_paths_in_safe(
	safe_dir: Union[str, Path],
	*,
	resolution_dir_hint: str = "R10m",
) -> Tuple[str, str]:
	"""Find Sentinel-2 red (B04) and NIR (B08) JP2 paths in a .SAFE folder.

	Expected layout (typical):
		<SAFE>.SAFE/GRANULE/<granule_id>/IMG_DATA/R10m/*.jp2

	This function is defensive and will also look for folders named "R10".
	Returns:
		(red_b04_path, nir_b08_path)
	"""

	safe_dir = Path(safe_dir)
	if not safe_dir.exists():
		raise FileNotFoundError(f"SAFE directory does not exist: {safe_dir}")
	if not safe_dir.is_dir():
		raise ValueError(f"Expected a SAFE directory path, got a file: {safe_dir}")

	granule_root = safe_dir / "GRANULE"
	if not granule_root.exists() or not granule_root.is_dir():
		raise ValueError(f"Not a Sentinel .SAFE folder (missing GRANULE/): {safe_dir}")

	# Collect candidates by granule, so we can detect ambiguity.
	# Sentinel-2 SAFE commonly has exactly one granule, but not always.
	granule_dirs = sorted([p for p in granule_root.iterdir() if p.is_dir()])
	if not granule_dirs:
		raise ValueError(f"No granules found under {granule_root}")

	def _find_in_img_data(granule_dir: Path) -> Dict[str, Path]:
		img_data_root = granule_dir / "IMG_DATA"
		if not img_data_root.exists():
			return {}

		# Prefer resolution hint, but also check a couple common alternatives.
		preferred = [resolution_dir_hint, "R10", "R10m"]
		search_roots: list[Path] = []
		for name in preferred:
			p = img_data_root / name
			if p.exists() and p.is_dir() and p not in search_roots:
				search_roots.append(p)

		# If none of the preferred roots exist, fall back to any directory with "R10" in its name.
		if not search_roots:
			for p in img_data_root.rglob("*"):
				if p.is_dir() and "R10" in p.name:
					search_roots.append(p)
			search_roots = sorted(set(search_roots))

		candidates: Dict[str, list[Path]] = {"B04": [], "B08": []}
		for root in search_roots:
			for jp2 in root.rglob("*.jp2"):
				name = jp2.name.upper()
				if "B04" in name:
					candidates["B04"].append(jp2)
				if "B08" in name:
					candidates["B08"].append(jp2)

		result: Dict[str, Path] = {}
		for band in ("B04", "B08"):
			paths = sorted(set(candidates[band]))
			if len(paths) == 1:
				result[band] = paths[0]
			elif len(paths) > 1:
				# If multiple, prefer 10m product naming if present (common: *_10m.jp2).
				preferred_paths = [p for p in paths if "_10M" in p.name.upper()]
				preferred_paths = sorted(preferred_paths)
				if len(preferred_paths) == 1:
					result[band] = preferred_paths[0]
				else:
					return {}
			else:
				return {}

		return result

	pairs: list[Tuple[Path, Path, Path]] = []
	for granule_dir in granule_dirs:
		found = _find_in_img_data(granule_dir)
		if "B04" in found and "B08" in found:
			pairs.append((granule_dir, found["B04"], found["B08"]))

	if not pairs:
		raise ValueError(
			"Could not find unique B04 and B08 JP2 files under GRANULE/*/IMG_DATA. "
			"Expected something like .../IMG_DATA/R10m/*B04*.jp2 and *B08*.jp2"
		)

	if len(pairs) > 1:
		granules = ", ".join([p[0].name for p in pairs[:5]])
		raise ValueError(
			f"Multiple granules contain B04/B08 pairs ({len(pairs)} found): {granules}. "
			"Please provide a SAFE with a single granule, or restructure input to target one granule."
		)

	_, red_path, nir_path = pairs[0]
	return str(red_path), str(nir_path)


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

