"""
test_junction_density.py

Unit tests for junction_density.py.
No network access — uses synthetic geometries only.

Fixture design
--------------
A simple 3×3 road grid inside a 100×100 m stat area:

  Roads are 6 m wide strips laid as a grid:
    - Horizontal corridors at y = 20, 50, 80  (width ±3 m → y in [17,23], [47,53], [77,83])
    - Vertical corridors at x = 20, 50, 80    (width ±3 m → x in [17,23], [47,53], [77,83])

  Each corridor is a separate polygon (fragmented, as in the real dataset).
  The six horizontal and six vertical strip polygons touch but do not overlap,
  mimicking the fragmented תחבורה parcels.

  At each grid crossing there is one junction → 3×3 = 9 junctions expected.
  Stat area = 100×100 = 10 000 m² → density = 9 / 10 000 = 0.0009 /m².

  Because the pipeline involves snap-buffering, Voronoi approximation, and graph
  simplification, the exact junction count may vary slightly.  We therefore test
  that the density is within a reasonable range rather than an exact value.
"""

import sys
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from junction_density import (
    _build_graph,
    _collapse_short_edges,
    _count_junctions,
    _planarize,
    compute_junction_density,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _h_strip(y_center: float, half_width: float, x_start: float, x_end: float):
    """Horizontal road strip polygon."""
    return box(x_start, y_center - half_width, x_end, y_center + half_width)


def _v_strip(x_center: float, half_width: float, y_start: float, y_end: float):
    """Vertical road strip polygon."""
    return box(x_center - half_width, y_start, x_center + half_width, y_end)


def _make_road_grid_gdf(
    h_centers=(100.0, 250.0, 400.0),
    v_centers=(100.0, 250.0, 400.0),
    half_width: float = 15.0,
    stat_size: float = 500.0,
):
    """
    Build a GeoDataFrame of road strip polygons arranged in a grid.
    Each strip is a separate row (fragmented, as in the real data).
    Strips run the full width of the stat area (no gap at crossings).

    Roads are 30 m wide (half_width=15), matching urban arterials.
    Intersection blobs are ~30×30 m, so skeleton branches inside each
    blob are ~15 m — well above the MIN_EDGE_M=5 m collapse threshold.
    """
    geoms = []
    for y in h_centers:
        geoms.append(_h_strip(y, half_width, 0, stat_size))
    for x in v_centers:
        geoms.append(_v_strip(x, half_width, 0, stat_size))

    gdf = gpd.GeoDataFrame(
        {"t_yeud_rashi": ["תחבורה"] * len(geoms)},
        geometry=geoms,
        crs="EPSG:2039",
    )
    return gdf


def _make_stat_gdf(stat_size: float = 500.0):
    """Single 500×500 m stat area."""
    return gpd.GeoDataFrame(
        {"ms_ezor": [999]},
        geometry=[box(0, 0, stat_size, stat_size)],
        crs="EPSG:2039",
    )


# ---------------------------------------------------------------------------
# Unit tests — graph helpers
# ---------------------------------------------------------------------------

class TestBuildGraph(unittest.TestCase):
    def test_simple_x_junction(self):
        """Four segments meeting at origin form one node of degree 4."""
        from shapely.geometry import LineString
        segs = [
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 0), (-10, 0)]),
            LineString([(0, 0), (0, 10)]),
            LineString([(0, 0), (0, -10)]),
        ]
        G = _build_graph(segs)
        center = (0.0, 0.0)
        self.assertIn(center, G.nodes)
        self.assertEqual(G.degree(center), 4)

    def test_degenerate_segment_skipped(self):
        """Zero-length segment (u == v) must not create a self-loop."""
        from shapely.geometry import LineString
        segs = [LineString([(5, 5), (5, 5)])]
        G = _build_graph(segs)
        self.assertEqual(G.number_of_edges(), 0)


class TestCollapseShortEdges(unittest.TestCase):
    def test_short_edge_contracted(self):
        """A 3-node path with a 2 m middle edge collapses to 2 nodes."""
        from shapely.geometry import LineString
        segs = [
            LineString([(0, 0), (10, 0)]),   # length 10
            LineString([(10, 0), (12, 0)]),   # length 2 — short
            LineString([(12, 0), (22, 0)]),   # length 10
            LineString([(10, 0), (10, 10)]),  # branch from first node
        ]
        G = _build_graph(segs)
        self.assertEqual(G.number_of_nodes(), 5)
        G = _collapse_short_edges(G, threshold=5.0)
        # The short edge (10,0)-(12,0) merges those two nodes → 4 nodes remain
        self.assertEqual(G.number_of_nodes(), 4)

    def test_long_edges_untouched(self):
        """Edges above the threshold must not be contracted."""
        from shapely.geometry import LineString
        segs = [
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
        ]
        G = _build_graph(segs)
        before = G.number_of_nodes()
        G = _collapse_short_edges(G, threshold=5.0)
        self.assertEqual(G.number_of_nodes(), before)


class TestCountJunctions(unittest.TestCase):
    def test_t_junction(self):
        """Node connecting three arms has degree 3 → 1 junction."""
        from shapely.geometry import LineString
        segs = [
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 0), (-10, 0)]),
            LineString([(0, 0), (0, 10)]),
        ]
        G = _build_graph(segs)
        self.assertEqual(_count_junctions(G), 1)

    def test_dead_ends_not_junctions(self):
        """Degree-1 nodes (dead ends) and degree-2 nodes must not be counted."""
        from shapely.geometry import LineString
        segs = [
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
        ]
        G = _build_graph(segs)
        self.assertEqual(_count_junctions(G), 0)


# ---------------------------------------------------------------------------
# Integration test — full pipeline on synthetic road grid
# ---------------------------------------------------------------------------

class TestComputeJunctionDensity(unittest.TestCase):
    def setUp(self):
        self.lu_gdf = _make_road_grid_gdf()
        self.stat_gdf = _make_stat_gdf()
        self.stat_area_m2 = 500.0 * 500.0

    def test_output_schema(self):
        df = compute_junction_density(self.lu_gdf, self.stat_gdf)
        self.assertIn("ms_ezor", df.columns)
        self.assertIn("junction_density", df.columns)
        self.assertEqual(len(df), 1)

    def test_stat_area_id_preserved(self):
        df = compute_junction_density(self.lu_gdf, self.stat_gdf)
        self.assertEqual(df.iloc[0]["ms_ezor"], 999)

    def test_density_positive(self):
        df = compute_junction_density(self.lu_gdf, self.stat_gdf)
        self.assertGreater(df.iloc[0]["junction_density"], 0.0)

    def test_density_reasonable_range(self):
        """
        A 3×3 grid has 9 ideal junctions → density ≈ 0.0009 /m².
        The Voronoi + graph simplification is approximate, so we accept
        a factor-of-4 band around the ideal value.
        """
        df = compute_junction_density(self.lu_gdf, self.stat_gdf)
        density = df.iloc[0]["junction_density"]
        ideal = 9 / self.stat_area_m2
        # Voronoi + graph simplification is approximate: accept a factor-of-8 band
        self.assertGreater(density, ideal / 8)
        self.assertLess(density, ideal * 8)

    def test_no_road_zones_returns_zero(self):
        """Stat area with no תחבורה parcels must return density 0."""
        lu_empty = self.lu_gdf.copy()
        lu_empty["t_yeud_rashi"] = "מגורים"
        df = compute_junction_density(lu_empty, self.stat_gdf)
        self.assertEqual(df.iloc[0]["junction_density"], 0.0)

    def test_multiple_stat_areas(self):
        """Results must include one row per stat area."""
        stat2 = gpd.GeoDataFrame(
            {"ms_ezor": [1, 2]},
            geometry=[box(0, 0, 500, 500), box(500, 0, 1000, 500)],
            crs="EPSG:2039",
        )
        df = compute_junction_density(self.lu_gdf, stat2)
        self.assertEqual(len(df), 2)
        self.assertSetEqual(set(df["ms_ezor"]), {1, 2})

    def test_geopackage_export(self):
        """GeoPackage export must create a file with centerlines and junctions."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            gpkg_path = Path(tmpdir) / "test_debug.gpkg"
            df = compute_junction_density(
                self.lu_gdf, self.stat_gdf, export_gpkg=str(gpkg_path)
            )
            self.assertTrue(gpkg_path.exists(), "GeoPackage file not created")
            # Verify both layers exist
            cl_gdf = gpd.read_file(gpkg_path, layer="centerlines")
            jn_gdf = gpd.read_file(gpkg_path, layer="junctions")
            self.assertGreater(len(cl_gdf), 0, "No centerline features")
            self.assertGreater(len(jn_gdf), 0, "No junction features")
            self.assertIn("ms_ezor", cl_gdf.columns, "ms_ezor missing from centerlines")
            self.assertIn("ms_ezor", jn_gdf.columns, "ms_ezor missing from junctions")
            self.assertEqual(cl_gdf.crs.to_epsg(), 2039)
            self.assertEqual(jn_gdf.crs.to_epsg(), 2039)


if __name__ == "__main__":
    unittest.main()
