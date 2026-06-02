"""
gis_preprocessing.py
Fetches Tel Aviv land-use (layer 514) and statistical area (layer 512) polygons
from the ArcGIS REST service, intersects them, and produces a flat CSV matching
the project schema:
    oid_migrash, ms_gush, ms_migrash, k_yeud_rashi, t_yeud_rashi,
    Shape_Length, Shape_Area, Shape, ms_ezor, ms_ezor_shetach
"""

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOGS_DIR / "gis_preprocessing.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

BASE_514 = "https://gisn.tel-aviv.gov.il/arcgis/rest/services/WM/IView2WM/MapServer/514/query"
BASE_512 = "https://gisn.tel-aviv.gov.il/arcgis/rest/services/WM/IView2WM/MapServer/512/query"

OUTPUT_COLUMNS = [
    "oid_migrash", "ms_gush", "ms_migrash",
    "k_yeud_rashi", "t_yeud_rashi",
    "Shape_Length", "Shape_Area", "Shape",
    "ms_ezor", "ms_ezor_shetach",
]

PAGE_SIZE = 2000
TARGET_CRS = "EPSG:2039"
MAX_WORKERS = 8


def _fetch_count(base_url: str) -> int:
    """Return total feature count for the layer."""
    resp = requests.get(
        base_url,
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["count"]


def _fetch_page(base_url: str, fields_str: str, offset: int) -> gpd.GeoDataFrame:
    """Fetch a single page and return a GeoDataFrame."""
    params = {
        "where": "1=1",
        "f": "geojson",
        "outFields": fields_str,
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
    }
    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    batch = gpd.read_file(io.StringIO(resp.text))
    log.debug("fetched %d features at offset %d from %s", len(batch), offset, base_url)
    return batch


def fetch_layer(base_url: str, out_fields: list[str]) -> gpd.GeoDataFrame:
    """Fetch a full ArcGIS REST layer using parallel page requests."""
    fields_str = ",".join(out_fields)

    total = _fetch_count(base_url)
    log.info("Total features to fetch from %s: %d", base_url, total)

    offsets = range(0, total, PAGE_SIZE)
    batches: list[gpd.GeoDataFrame] = [None] * len(offsets)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_page, base_url, fields_str, offset): i
            for i, offset in enumerate(offsets)
        }
        for future in as_completed(futures):
            i = futures[future]
            batches[i] = future.result()

    return pd.concat(batches, ignore_index=True) if batches else gpd.GeoDataFrame()


def load_land_use() -> gpd.GeoDataFrame:
    """Fetch layer 514 — land-use parcels."""
    log.info("Fetching land-use layer (514)...")
    fields = ["oid_migrash", "ms_gush", "ms_migrash", "k_yeud_rashi", "t_yeud_rashi"]
    gdf = fetch_layer(BASE_514, fields)
    log.info("Land-use layer: %d features", len(gdf))
    gdf = gdf.to_crs(TARGET_CRS)
    return gdf


def load_stat_areas() -> gpd.GeoDataFrame:
    """Fetch layer 512 — statistical areas. Shape_Area renamed to ms_ezor_shetach."""
    log.info("Fetching statistical areas layer (512)...")
    fields = ["ms_ezor", "Shape_Area"]
    gdf = fetch_layer(BASE_512, fields)
    log.info("Stat areas layer: %d features", len(gdf))
    gdf = gdf.rename(columns={"Shape_Area": "ms_ezor_shetach"})
    gdf = gdf.to_crs(TARGET_CRS)
    return gdf


def intersect_layers(lu: gpd.GeoDataFrame, stat: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Intersect land-use with statistical areas and recompute geometry fields."""
    log.info("Intersecting %d LU features with %d stat areas...", len(lu), len(stat))
    result = gpd.overlay(lu, stat, how="intersection", keep_geom_type=True)

    before = len(result)
    # Drop parcels that fell outside every statistical area
    result = result.dropna(subset=["ms_ezor"]).copy()
    dropped = before - len(result)
    if dropped:
        log.warning("Dropped %d features with no matching stat area", dropped)
    log.info("Intersection result: %d features", len(result))

    # Recompute geometry-derived fields from the intersection result (values in m²)
    result["Shape_Area"] = result.geometry.area
    result["Shape_Length"] = result.geometry.length
    result["Shape"] = "Polygon"

    return result


def build_csv(output_path: str = "resources/yk_tlv_stat.csv") -> gpd.GeoDataFrame:
    """Full pipeline: fetch → intersect → export CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lu = load_land_use()
    stat = load_stat_areas()
    result = intersect_layers(lu, stat)

    out = result[OUTPUT_COLUMNS].reset_index(drop=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("Saved %d rows to %s", len(out), output_path)
    return out


if __name__ == "__main__":
    build_csv()
