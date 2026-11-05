import math
import numpy as np
from typing import Any
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from d_temporal_moe.evaluation.metrics import score

PROTOCOL = "Native30m-v1"
EXPECTED_VALID_PIXELS = 1959211
EXPECTED_CDL_SHA256 = "100f0b664367453ec38e28e1e72812af66b6308f29b5c2f166acdb708940b922"
SOURCE_CRS_WKT = 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","3857"]]'
TARGET_CRS_WKT = 'PROJCS["NAD83 / Conus Albers",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","5070"]]'


def native_arrays(
    probability: np.ndarray, source_transform: tuple[float, ...], cdl: Any
) -> dict[str, Any]:
    if probability.shape != (256, 256):
        raise ValueError(f"expected 256x256 probability, got {probability.shape}")
    source = Affine(*map(float, source_transform[:6]))
    bounds = transform_bounds(
        SOURCE_CRS_WKT, TARGET_CRS_WKT, *array_bounds(256, 256, source), densify_pts=41
    )
    float_window = from_bounds(*bounds, cdl.transform)
    col0 = math.floor(float_window.col_off) - 1
    row0 = math.floor(float_window.row_off) - 1
    window = Window(
        col0,
        row0,
        math.ceil(float_window.col_off + float_window.width) - col0 + 1,
        math.ceil(float_window.row_off + float_window.height) - row0 + 1,
    )
    code = cdl.read(1, window=window)
    target_transform = cdl.window_transform(window)
    source_padded = np.pad(probability.astype(np.float32), 8)
    coverage_source = np.pad(np.ones((256, 256), dtype=np.float32), 8)
    source_shifted = source * Affine.translation(-8, -8)
    destination = np.zeros(code.shape, dtype=np.float32)
    coverage = np.zeros(code.shape, dtype=np.float32)
    reproject(
        source_padded,
        destination,
        src_transform=source_shifted,
        src_crs=SOURCE_CRS_WKT,
        dst_transform=target_transform,
        dst_crs=TARGET_CRS_WKT,
        resampling=Resampling.average,
        src_nodata=None,
        dst_nodata=0,
        ERROR_THRESHOLD=0.0,
    )
    reproject(
        coverage_source,
        coverage,
        src_transform=source_shifted,
        src_crs=SOURCE_CRS_WKT,
        dst_transform=target_transform,
        dst_crs=TARGET_CRS_WKT,
        resampling=Resampling.average,
        src_nodata=None,
        dst_nodata=0,
        ERROR_THRESHOLD=0.0,
    )
    valid = (code != 0) & (coverage >= 1.0 - 1e-06)
    target = np.where(valid, (code == 1).astype(np.uint8), 255).astype(np.uint8)
    prediction = (destination >= 0.5).astype(np.uint8)
    rows, cols = np.indices(code.shape)
    ids = ((row0 + rows) * cdl.width + (col0 + cols)).astype(np.int64)
    return {
        "prediction": prediction,
        "target": target,
        "ids": ids,
        "valid": valid,
        "source_transform": tuple(source)[:6],
        "target_transform": tuple(target_transform)[:6],
        "coverage": coverage,
        "window": [int(row0), int(col0), int(window.height), int(window.width)],
    }


def _deduplicate(
    ids: np.ndarray, prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _, first = np.unique(ids, return_index=True)
    first = np.sort(first)
    return (prediction[first], target[first])
