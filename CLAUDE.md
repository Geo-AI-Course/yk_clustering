# CLAUDE

## Project Goal

Cluster Tel Aviv statistical areas by land-use zoning composition and spatial context to derive urban typologies, with a later goal of classifying new urban plans by similarity to existing clusters.

## Data Sources

- `yk_tlv.csv`: land-use zoning table for Tel Aviv.
- External GeoJSON / GIS layers required:
  - land-use polygons with zoning categories
  - Tel Aviv statistical area polygons

## Pipeline Steps

1. Load geometries
   - import land-use polygon layer
   - import statistical area polygon layer
2. Spatial join
   - assign each land-use polygon to a statistical area
   - use polygon containment or intersection logic
3. Aggregate features per statistical area
   - compute total area per land-use category within each statistical area
   - compute normalized percentage of area by category
4. Build feature matrix
   - one row per statistical area
   - features = percentage area per land-use type
5. Run clustering
   - use baseline clustering such as KMeans
   - select cluster count by silhouette, elbow, or domain validation
6. Interpret clusters
   - label clusters as residential, commercial, mixed-use, industrial, etc.
   - evaluate cluster profiles by dominant land-use percentages

## Baseline Feature Engineering

- Simple and interpretable features only:
  - percentage of total area per land-use type in each statistical area
- Do not add complex spatial metrics in the MVP
- Keep the feature dimension bounded by land-use categories

## Constraints and Design Notes

- Dataset is small: hundreds of statistical areas, not tens of thousands of samples
- Current CSV lacks geometry, so spatial join depends on external GeoJSON/GIS layers
- Avoid overly complex spatial features for the first version
- Focus on reproducible area-based composition features

## Suggested Next Improvements

- Add additional spatial context after baseline clustering:
  - adjacency/connectivity between statistical areas
  - density measures or fragmentation metrics
  - distance to major land-use anchors or transport nodes
- Incorporate derived contextual features only if baseline clusters need refinement
- Explore supervised similarity/classification of new plans once stable clusters exist
- Validate against known planning typologies and local domain knowledge
