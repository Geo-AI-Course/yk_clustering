"""
test_gis_preprocessing.py
Unit tests for gis_preprocessing.py.
All HTTP calls are mocked — no network access required.
"""

import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, mapping

# ---------------------------------------------------------------------------
# Helpers to build minimal GeoJSON FeatureCollections
# ---------------------------------------------------------------------------

def _make_geojson(features: list[dict]) -> str:
    return json.dumps({"type": "FeatureCollection", "features": features})


def _polygon_feature(coords, props: dict) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": props,
    }


# Six non-overlapping unit squares across three stat areas (planar EPSG:2039 coords)
SQUARE_A = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]   # stat 5
SQUARE_B = [(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)]   # stat 5
SQUARE_C = [(2, 0), (3, 0), (3, 1), (2, 1), (2, 0)]   # stat 6
SQUARE_D = [(3, 0), (4, 0), (4, 1), (3, 1), (3, 0)]   # stat 6
SQUARE_E = [(0, 1), (1, 1), (1, 2), (0, 2), (0, 1)]   # stat 7
SQUARE_F = [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]   # stat 7

STAT_AREA_1 = [(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)]  # ms_ezor=5, area=2
STAT_AREA_2 = [(2, 0), (4, 0), (4, 1), (2, 1), (2, 0)]  # ms_ezor=6, area=2
STAT_AREA_3 = [(0, 1), (2, 1), (2, 2), (0, 2), (0, 1)]  # ms_ezor=7, area=2
STAT_AREA_VALUE = 2.0  # area of each stat polygon (planar units = m² in EPSG:2039)

LU_GEOJSON = _make_geojson([
    _polygon_feature(SQUARE_A, {"oid_migrash": 1, "ms_gush": 100, "ms_migrash": 10,
                                 "k_yeud_rashi": 1001, "t_yeud_rashi": "מגורים"}),
    _polygon_feature(SQUARE_B, {"oid_migrash": 2, "ms_gush": 100, "ms_migrash": 11,
                                 "k_yeud_rashi": 1002, "t_yeud_rashi": "מסחר"}),
    _polygon_feature(SQUARE_C, {"oid_migrash": 3, "ms_gush": 101, "ms_migrash": 20,
                                 "k_yeud_rashi": 1001, "t_yeud_rashi": "מגורים"}),
    _polygon_feature(SQUARE_D, {"oid_migrash": 4, "ms_gush": 101, "ms_migrash": 21,
                                 "k_yeud_rashi": 1003, "t_yeud_rashi": "תעסוקה"}),
    _polygon_feature(SQUARE_E, {"oid_migrash": 5, "ms_gush": 102, "ms_migrash": 30,
                                 "k_yeud_rashi": 1005, "t_yeud_rashi": "מבנים ומוסדות ציבור"}),
    _polygon_feature(SQUARE_F, {"oid_migrash": 6, "ms_gush": 102, "ms_migrash": 31,
                                 "k_yeud_rashi": 1006, "t_yeud_rashi": "שטחים פתוחים"}),
])

STAT_GEOJSON = _make_geojson([
    _polygon_feature(STAT_AREA_1, {"ms_ezor": 5, "Shape_Area": STAT_AREA_VALUE}),
    _polygon_feature(STAT_AREA_2, {"ms_ezor": 6, "Shape_Area": STAT_AREA_VALUE}),
    _polygon_feature(STAT_AREA_3, {"ms_ezor": 7, "Shape_Area": STAT_AREA_VALUE}),
])


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


def _mock_count_response(count: int) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"count": count}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import gis_preprocessing as gp  # noqa: E402  (after sys.path insert)


class TestFetchLayer(unittest.TestCase):
    """fetch_layer — parallel fetch logic."""

    def test_single_page(self):
        """Returns all features when total fits in one page."""
        with patch("gis_preprocessing._fetch_count", return_value=6), \
             patch("gis_preprocessing._fetch_page",
                   return_value=gpd.read_file(io.StringIO(LU_GEOJSON))) as mock_page:
            fields = ["oid_migrash", "ms_gush", "ms_migrash", "k_yeud_rashi", "t_yeud_rashi"]
            gdf = gp.fetch_layer(gp.BASE_514, fields)

        self.assertEqual(len(gdf), 6)
        self.assertIn("oid_migrash", gdf.columns)
        mock_page.assert_called_once_with(gp.BASE_514, ",".join(fields), 0)

    def test_multiple_pages_all_fetched(self):
        """All pages are fetched when total > PAGE_SIZE."""
        total = gp.PAGE_SIZE + 6
        full_page = gpd.read_file(io.StringIO(_make_geojson([
            _polygon_feature(SQUARE_A, {"oid_migrash": i, "ms_gush": 1,
                                         "ms_migrash": i, "k_yeud_rashi": 1001,
                                         "t_yeud_rashi": "מגורים"})
            for i in range(gp.PAGE_SIZE)
        ])))
        last_page = gpd.read_file(io.StringIO(LU_GEOJSON))

        def fake_fetch_page(url, fields_str, offset):
            return full_page if offset == 0 else last_page

        with patch("gis_preprocessing._fetch_count", return_value=total), \
             patch("gis_preprocessing._fetch_page", side_effect=fake_fetch_page):
            fields = ["oid_migrash", "ms_gush", "ms_migrash", "k_yeud_rashi", "t_yeud_rashi"]
            gdf = gp.fetch_layer(gp.BASE_514, fields)

        self.assertEqual(len(gdf), total)

    def test_empty_service_returns_empty_geodataframe(self):
        with patch("gis_preprocessing._fetch_count", return_value=0):
            gdf = gp.fetch_layer(gp.BASE_514, ["oid_migrash"])

        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertEqual(len(gdf), 0)


class TestLoadLandUse(unittest.TestCase):
    """load_land_use — correct fields fetched and CRS set."""

    def test_returns_target_crs(self):
        with patch("gis_preprocessing.fetch_layer") as mock_fetch:
            gdf_raw = gpd.read_file(io.StringIO(LU_GEOJSON))
            mock_fetch.return_value = gdf_raw
            result = gp.load_land_use()

        self.assertEqual(result.crs.to_epsg(), 2039)

    def test_requested_fields(self):
        with patch("gis_preprocessing.fetch_layer") as mock_fetch:
            gdf_raw = gpd.read_file(io.StringIO(LU_GEOJSON))
            mock_fetch.return_value = gdf_raw
            gp.load_land_use()

        _, call_kwargs = mock_fetch.call_args
        # fetch_layer is called positionally
        requested_fields = mock_fetch.call_args[0][1]
        self.assertIn("oid_migrash", requested_fields)
        self.assertIn("k_yeud_rashi", requested_fields)
        self.assertIn("t_yeud_rashi", requested_fields)


class TestLoadStatAreas(unittest.TestCase):
    """load_stat_areas — Shape_Area renamed and CRS set."""

    def test_shape_area_renamed_to_ms_ezor_shetach(self):
        with patch("gis_preprocessing.fetch_layer") as mock_fetch:
            gdf_raw = gpd.read_file(io.StringIO(STAT_GEOJSON))
            mock_fetch.return_value = gdf_raw
            result = gp.load_stat_areas()

        self.assertIn("ms_ezor_shetach", result.columns)
        self.assertNotIn("Shape_Area", result.columns)

    def test_returns_target_crs(self):
        with patch("gis_preprocessing.fetch_layer") as mock_fetch:
            gdf_raw = gpd.read_file(io.StringIO(STAT_GEOJSON))
            mock_fetch.return_value = gdf_raw
            result = gp.load_stat_areas()

        self.assertEqual(result.crs.to_epsg(), 2039)

    def test_ms_ezor_shetach_value_preserved(self):
        with patch("gis_preprocessing.fetch_layer") as mock_fetch:
            gdf_raw = gpd.read_file(io.StringIO(STAT_GEOJSON))
            mock_fetch.return_value = gdf_raw
            result = gp.load_stat_areas()

        # load_stat_areas preserves the raw server value; /1000 happens in intersect_layers
        self.assertAlmostEqual(float(result["ms_ezor_shetach"].iloc[0]), STAT_AREA_VALUE)


class TestIntersectLayers(unittest.TestCase):
    """intersect_layers — geometry recomputed, nulls dropped, schema correct."""

    def _make_inputs(self):
        lu = gpd.read_file(io.StringIO(LU_GEOJSON)).set_crs("EPSG:2039", allow_override=True)
        stat = gpd.read_file(io.StringIO(STAT_GEOJSON)).set_crs("EPSG:2039", allow_override=True)
        stat = stat.rename(columns={"Shape_Area": "ms_ezor_shetach"})
        return lu, stat

    def test_output_columns_present(self):
        lu, stat = self._make_inputs()
        result = gp.intersect_layers(lu, stat)
        for col in gp.OUTPUT_COLUMNS:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_no_null_ms_ezor(self):
        lu, stat = self._make_inputs()
        result = gp.intersect_layers(lu, stat)
        self.assertFalse(result["ms_ezor"].isna().any())

    def test_shape_column_is_polygon(self):
        lu, stat = self._make_inputs()
        result = gp.intersect_layers(lu, stat)
        self.assertTrue((result["Shape"] == "Polygon").all())

    def test_shape_area_is_positive(self):
        lu, stat = self._make_inputs()
        result = gp.intersect_layers(lu, stat)
        self.assertTrue((result["Shape_Area"] > 0).all())

    def test_intersection_area_le_stat_area(self):
        lu, stat = self._make_inputs()
        result = gp.intersect_layers(lu, stat)
        # both divided by 1000 — inequality must still hold
        self.assertTrue((result["Shape_Area"] <= result["ms_ezor_shetach"] + 1e-9).all())

    def test_areas_in_thousands_of_sqm(self):
        """ms_ezor_shetach must be >= Shape_Area for every row."""
        lu, stat = self._make_inputs()
        result = gp.intersect_layers(lu, stat)
        self.assertTrue((result["ms_ezor_shetach"] >= result["Shape_Area"]).all())

    def test_row_count(self):
        """All 6 LU parcels each fully within one stat area → 6 rows."""
        lu, stat = self._make_inputs()
        result = gp.intersect_layers(lu, stat)
        self.assertEqual(len(result), 6)

    def test_parcels_outside_stat_areas_dropped(self):
        """A LU parcel entirely outside all stat areas must be dropped."""
        lu, stat = self._make_inputs()
        # Add a parcel far outside the stat area polygon
        outside = _polygon_feature(
            [(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)],
            {"oid_migrash": 99, "ms_gush": 999, "ms_migrash": 99,
             "k_yeud_rashi": 1001, "t_yeud_rashi": "מגורים"},
        )
        lu_extra = gpd.read_file(
            io.StringIO(_make_geojson(json.loads(LU_GEOJSON)["features"] + [outside]))
        ).set_crs("EPSG:2039", allow_override=True)

        result = gp.intersect_layers(lu_extra, stat)
        self.assertFalse((result["oid_migrash"] == 99).any())


RESULTS_DIR = Path(__file__).resolve().parent / "results"


class TestBuildCsv(unittest.TestCase):
    """build_csv — integration: correct CSV written to tests/results/."""

    OUTPUT_PATH = str(RESULTS_DIR / "yk_tlv_stat_test.csv")

    @classmethod
    def setUpClass(cls):
        RESULTS_DIR.mkdir(exist_ok=True)

    def _patched_load(self, mock_load_lu, mock_load_stat):
        lu = gpd.read_file(io.StringIO(LU_GEOJSON)).set_crs("EPSG:2039", allow_override=True)
        stat = gpd.read_file(io.StringIO(STAT_GEOJSON)).set_crs("EPSG:2039", allow_override=True)
        stat = stat.rename(columns={"Shape_Area": "ms_ezor_shetach"})
        mock_load_lu.return_value = lu
        mock_load_stat.return_value = stat

    def test_csv_columns_match_schema(self):
        with patch("gis_preprocessing.load_land_use") as mlu, \
             patch("gis_preprocessing.load_stat_areas") as mstat:
            self._patched_load(mlu, mstat)
            gp.build_csv(output_path=self.OUTPUT_PATH)

        df = pd.read_csv(self.OUTPUT_PATH, encoding="utf-8-sig")
        self.assertEqual(list(df.columns), gp.OUTPUT_COLUMNS)

    def test_csv_has_rows(self):
        with patch("gis_preprocessing.load_land_use") as mlu, \
             patch("gis_preprocessing.load_stat_areas") as mstat:
            self._patched_load(mlu, mstat)
            gp.build_csv(output_path=self.OUTPUT_PATH)

        df = pd.read_csv(self.OUTPUT_PATH, encoding="utf-8-sig")
        self.assertGreater(len(df), 0)

    def test_csv_hebrew_readable(self):
        """Hebrew text in t_yeud_rashi must survive the CSV round-trip."""
        with patch("gis_preprocessing.load_land_use") as mlu, \
             patch("gis_preprocessing.load_stat_areas") as mstat:
            self._patched_load(mlu, mstat)
            gp.build_csv(output_path=self.OUTPUT_PATH)

        df = pd.read_csv(self.OUTPUT_PATH, encoding="utf-8-sig")
        hebrew_values = df["t_yeud_rashi"].dropna().tolist()
        self.assertTrue(any("מגורים" in v for v in hebrew_values))


if __name__ == "__main__":
    unittest.main()
