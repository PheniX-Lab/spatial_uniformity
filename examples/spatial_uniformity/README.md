# Spatial uniformity examples

These examples contain projected hill coordinates for three field plots
from the 2023 study: 128, 113 and 313. They have similar hill densities and
a gradient in spatial uniformity. Images are not required for this calculation.

## Files and coordinate system

- `points/`: three hill-coordinate CSVs, with an `X,Y` header.
- `boundaries.csv`: a combined table without a header, with columns
  `plot_id,x,y` and four ordered analysis-window vertices per plot.
- `expected_results.csv`: reference counts, area, density, CV and uniformity.

The hill coordinates and boundary vertices are copied unchanged from the
same projected planar coordinate system in the original dataset. With
metre-based coordinates, areas are in square metres and densities in hills/m2.
The supplied coordinates are independent of the P2PNet test images in
`../p2pnet/`; do not combine those pixel predictions with these boundaries.

## Calculation

All supplied hills define the Voronoi tessellation, including neighbouring
hills outside the analysis window. Only complete, finite Voronoi polygons
entirely inside the selected window enter the area statistics. Unbounded
and boundary-crossing cells are excluded; polygons are not clipped.
The window must be convex, with vertices ordered clockwise or counterclockwise.

Spatial uniformity is `1 - CV`, where `CV = sample SD(area) / mean(area)`
and sample SD uses `ddof=1`. A lower CV corresponds to higher uniformity.
Input row counts, hills inside the window and retained cell counts describe
different quantities.

| Plot | Input hills | Hills inside window | Density | Retained cells | CV | Uniformity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 360 | 329 | 20.618 | 287 | 0.256203 | 0.743797 |
| 113 | 347 | 318 | 20.014 | 276 | 0.200997 | 0.799003 |
| 313 | 322 | 320 | 20.525 | 250 | 0.108659 | 0.891341 |

## Run

Run each command from the repository root:

```bash
python spatial_uniformity.py --points examples/spatial_uniformity/points/plot128_DSC00788.csv --boundary examples/spatial_uniformity/boundaries.csv --plot-id 128 --output-dir output/uniformity_examples
```

```bash
python spatial_uniformity.py --points examples/spatial_uniformity/points/plot113_DSC00931.csv --boundary examples/spatial_uniformity/boundaries.csv --plot-id 113 --output-dir output/uniformity_examples
```

```bash
python spatial_uniformity.py --points examples/spatial_uniformity/points/plot313_DSC01086.csv --boundary examples/spatial_uniformity/boundaries.csv --plot-id 313 --output-dir output/uniformity_examples
```

Each command writes a per-hill area/status CSV and a summary JSON.
Compare the JSON values with `expected_results.csv`. Add `--plot` to save
a Voronoi visualization (requires Matplotlib).

For other datasets, provide hill coordinates and boundary vertices in the
same projected planar coordinate system. Pixel-to-ground coordinate
conversion is a separate preprocessing step and is not implemented here.
