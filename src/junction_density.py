"""
junction_density.py

Computes junction density per statistical area from תחבורה (transport/road)
zone polygons.

Pipeline per statistical area:
  1. Clip תחבורה parcels to the stat area boundary
  2. Dissolve with a small snap buffer to close fragmentation gaps between parcels
  3. Compute a Voronoi-based centerline skeleton for each dissolved polygon part
  4. Planarize the centerline segments (split at mutual crossings)
  5. Build an undirected graph from the segments
  6. Collapse edges shorter than MIN_EDGE_M to remove intersection hairballs
     that arise from wide road corridors (Option B graph simplification)
  7. Count branch nodes (degree >= 3) as junctions
  8. junction_density = junction_count / stat_area_m²
"""

import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SNAP_BUFFER_M: float = 0.5   # metres — closes hairline gaps between touching fragments
MIN_EDGE_M: float = 5.0      # metres — edges shorter than this are collapsed in the graph
INTERP_DISTANCE: float = 1.0  # metres — boundary sampling density for Voronoi skeleton

ROAD_ZONE_LABEL = "תחבורה"
TARGET_CRS = "EPSG:2039"
OUTPUT_COLUMNS = ["ms_ezor", "junction_density"]


# ---------------------------------------------------------------------------
# Centerline (Voronoi skeleton)
# ---------------------------------------------------------------------------

def _densify_ring(ring, distance: float) -> np.ndarray:
    """Sample points at *distance* intervals along a LinearRing/LineString."""
    length = ring.length
    if length == 0:
        return np.empty((0, 2))
    n = max(int(length / distance), 4)
    pts = [ring.interpolate(i / n, normalized=True) for i in range(n)]
    return np.array([(p.x, p.y) for p in pts])


def _centerlines_for_polygon(polygon, interpolation_distance: float = INTERP_DISTANCE) -> list:
    """
    Return a list of LineString objects approximating the centerline skeleton
    of *polygon* using a Voronoi diagram on densified boundary points.

    Voronoi ridges whose midpoint lies inside the polygon are the skeleton.
    """
    try:
        from scipy.spatial import Voronoi  # local import to keep top-level lightweight

        pts = _densify_ring(polygon.exterior, interpolation_distance)
        for interior in polygon.interiors:
            pts = np.vstack([pts, _densify_ring(interior, interpolation_distance)])

        if len(pts) < 4:
            return []

        vor = Voronoi(pts)
        lines = []
        for ridge in vor.ridge_vertices:
            if -1 in ridge:          # infinite ridge — skip
                continue
            p1 = vor.vertices[ridge[0]]
            p2 = vor.vertices[ridge[1]]
            midpoint = Point((p1 + p2) / 2)
            if polygon.contains(midpoint):
                lines.append(LineString([p1, p2]))
        return lines

    except Exception as exc:
        log.debug("Centerline failed for polygon (area=%.1f): %s", polygon.area, exc)
        return []


# ---------------------------------------------------------------------------
# Graph construction and simplification
# ---------------------------------------------------------------------------

def _planarize(lines: list) -> list:
    """
    Split all lines at every mutual crossing via GEOS unary_union noding.
    Returns a flat list of LineString segments.
    """
    if not lines:
        return []
    merged = unary_union(lines)
    if merged.is_empty:
        return []
    if merged.geom_type == "LineString":
        return [merged]
    if merged.geom_type == "MultiLineString":
        return list(merged.geoms)
    return []


def _round_node(coord, decimals: int = 4) -> tuple:
    """Round coordinate to *decimals* places (0.1 mm precision in ITM metres)."""
    return (round(coord[0], decimals), round(coord[1], decimals))


def _build_graph(segments: list) -> nx.Graph:
    """
    Build an undirected graph from line segments.
    Nodes are rounded endpoint coordinates; parallel edges keep the shorter length.
    """
    G: nx.Graph = nx.Graph()
    for seg in segments:
        coords = list(seg.coords)
        u = _round_node(coords[0])
        v = _round_node(coords[-1])
        if u == v:
            continue
        length = seg.length
        if G.has_edge(u, v):
            if G[u][v]["length"] > length:
                G[u][v]["length"] = length
        else:
            G.add_edge(u, v, length=length)
    return G


def _collapse_short_edges(G: nx.Graph, threshold: float) -> nx.Graph:
    """
    Iteratively contract every edge shorter than *threshold*:
    merge the two endpoint nodes into one, inheriting all connections.

    This removes the micro-topology hairballs that appear at wide intersections
    (where the Voronoi skeleton produces many short branches near the junction
    center instead of a single clean branch point).
    """
    changed = True
    while changed:
        changed = False
        for u, v, data in list(G.edges(data=True)):
            if data.get("length", 0.0) >= threshold:
                continue
            if not (G.has_node(u) and G.has_node(v)) or u == v:
                continue
            # Merge v into u: transfer every edge from v to u
            for neighbor in list(G.neighbors(v)):
                if neighbor == u:
                    continue
                edge_len = G[v][neighbor].get("length", 0.0)
                if G.has_edge(u, neighbor):
                    if G[u][neighbor]["length"] > edge_len:
                        G[u][neighbor]["length"] = edge_len
                else:
                    G.add_edge(u, neighbor, length=edge_len)
            G.remove_node(v)
            changed = True
            break  # restart — node set has changed
    return G


def _count_junctions(G: nx.Graph) -> int:
    """Nodes with degree >= 3 are genuine intersections (T, X, Y, etc.)."""
    return sum(1 for _, deg in G.degree() if deg >= 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_junction_density(
    lu_gdf: gpd.GeoDataFrame,
    stat_gdf: gpd.GeoDataFrame,
    snap_buffer_m: float = SNAP_BUFFER_M,
    min_edge_m: float = MIN_EDGE_M,
    interpolation_distance: float = INTERP_DISTANCE,
    export_gpkg: str | None = None,
) -> pd.DataFrame:
    """
    Compute junction density per statistical area.

    Parameters
    ----------
    lu_gdf : GeoDataFrame
        Raw land-use polygons with ``t_yeud_rashi`` and geometry (EPSG:2039).
    stat_gdf : GeoDataFrame
        Statistical areas with ``ms_ezor`` and geometry (EPSG:2039).
    snap_buffer_m : float
        Buffer distance (metres) to close fragmentation gaps before dissolving.
    min_edge_m : float
        Edges shorter than this (metres) are collapsed during graph simplification.
    interpolation_distance : float
        Boundary sampling spacing (metres) for the Voronoi centerline skeleton.
    export_gpkg : str or None
        If provided, writes two layers to a GeoPackage at this path:
        ``centerlines`` (LineString) and ``junctions`` (Point), both with a
        ``ms_ezor`` attribute.  Suitable for visual inspection in ArcGIS Pro.

    Returns
    -------
    pd.DataFrame
        Columns: ``ms_ezor``, ``junction_density`` (junctions per m²).
    """
    if lu_gdf.crs and lu_gdf.crs.to_epsg() != 2039:
        lu_gdf = lu_gdf.to_crs(TARGET_CRS)
    if stat_gdf.crs and stat_gdf.crs.to_epsg() != 2039:
        stat_gdf = stat_gdf.to_crs(TARGET_CRS)

    roads = lu_gdf[lu_gdf["t_yeud_rashi"] == ROAD_ZONE_LABEL].copy()
    log.info("תחבורה parcels: %d", len(roads))

    records = []
    centerline_rows: list[dict] = []
    junction_rows: list[dict] = []
    total = len(stat_gdf)

    for i, (_, stat_row) in enumerate(stat_gdf.iterrows(), 1):
        ms_ezor = stat_row["ms_ezor"]
        stat_geom = stat_row.geometry
        stat_area_m2 = stat_geom.area
        log.debug("[%d/%d] stat area %s  (%.0f m²)", i, total, ms_ezor, stat_area_m2)

        # Step 1: candidate roads that touch this stat area
        candidate_mask = roads.geometry.intersects(stat_geom)
        roads_in = roads.loc[candidate_mask].copy()
        roads_in = roads_in.assign(geometry=roads_in.geometry.intersection(stat_geom))
        roads_in = roads_in[~roads_in.geometry.is_empty]

        if roads_in.empty:
            records.append({"ms_ezor": ms_ezor, "junction_density": 0.0})
            continue

        # Step 2: dissolve with snap buffer to merge touching/overlapping fragments
        dissolved = (
            roads_in.geometry
            .buffer(snap_buffer_m)
            .union_all()
            .buffer(-snap_buffer_m)
        )
        if dissolved is None or dissolved.is_empty:
            records.append({"ms_ezor": ms_ezor, "junction_density": 0.0})
            continue

        if dissolved.geom_type == "Polygon":
            parts = [dissolved]
        elif dissolved.geom_type == "MultiPolygon":
            parts = list(dissolved.geoms)
        else:
            records.append({"ms_ezor": ms_ezor, "junction_density": 0.0})
            continue

        # Step 3: Voronoi centerlines per polygon part
        all_lines = []
        for part in parts:
            if part.is_empty or part.area < 1.0:
                continue
            all_lines.extend(_centerlines_for_polygon(part, interpolation_distance))

        if not all_lines:
            records.append({"ms_ezor": ms_ezor, "junction_density": 0.0})
            continue

        # Steps 4–7: planarize → graph → collapse short edges → count junctions
        segments = _planarize(all_lines)
        G = _build_graph(segments)
        G = _collapse_short_edges(G, min_edge_m)
        junctions = _count_junctions(G)

        density = junctions / stat_area_m2 if stat_area_m2 > 0 else 0.0
        log.debug("  → %d junctions, density=%.6f /m²", junctions, density)
        records.append({"ms_ezor": ms_ezor, "junction_density": density})

        # Accumulate debug geometry if export requested
        if export_gpkg is not None:
            for seg in segments:
                centerline_rows.append({"ms_ezor": ms_ezor, "geometry": seg})
            for node, deg in G.degree():
                if deg >= 3:
                    junction_rows.append({"ms_ezor": ms_ezor, "geometry": Point(node)})

    log.info("Junction density computed for %d stat areas", len(records))

    if export_gpkg is not None:
        _write_debug_geopackage(export_gpkg, centerline_rows, junction_rows)

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def _write_debug_geopackage(
    path: str,
    centerline_rows: list[dict],
    junction_rows: list[dict],
) -> None:
    """Write centerlines and junction points to a GeoPackage for ArcGIS Pro."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if centerline_rows:
        cl_gdf = gpd.GeoDataFrame(centerline_rows, crs=TARGET_CRS)
        cl_gdf.to_file(out, layer="centerlines", driver="GPKG")
        log.info("Wrote %d centerline segments to '%s' (layer: centerlines)",
                 len(cl_gdf), out)
    else:
        log.warning("No centerline features to export.")

    if junction_rows:
        jn_gdf = gpd.GeoDataFrame(junction_rows, crs=TARGET_CRS)
        jn_gdf.to_file(out, layer="junctions", driver="GPKG")
        log.info("Wrote %d junction points to '%s' (layer: junctions)",
                 len(jn_gdf), out)
    else:
        log.warning("No junction features to export.")


def build_junction_density_csv(
    output_path: str = "resources/junction_density.csv",
    snap_buffer_m: float = SNAP_BUFFER_M,
    min_edge_m: float = MIN_EDGE_M,
    export_gpkg: str | None = "resources/junction_debug.gpkg",
) -> pd.DataFrame:
    """
    Fetch land-use and statistical area layers from the ArcGIS REST service,
    compute junction density, and save the result to *output_path*.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gis_preprocessing import BASE_512, BASE_514, fetch_layer

    log.info("Fetching land-use layer (514) for junction density...")
    lu_fields = ["oid_migrash", "ms_gush", "ms_migrash", "k_yeud_rashi", "t_yeud_rashi"]
    lu_gdf = fetch_layer(BASE_514, lu_fields)
    lu_gdf = lu_gdf.to_crs(TARGET_CRS)

    log.info("Fetching statistical areas layer (512)...")
    stat_fields = ["ms_ezor", "Shape_Area"]
    stat_gdf = fetch_layer(BASE_512, stat_fields)
    stat_gdf = stat_gdf.to_crs(TARGET_CRS)

    df = compute_junction_density(
        lu_gdf, stat_gdf, snap_buffer_m, min_edge_m, export_gpkg=export_gpkg
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("Saved %d rows to %s", len(df), output_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    build_junction_density_csv()
