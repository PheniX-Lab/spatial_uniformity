"""Calculate rice spatial uniformity from complete Voronoi cell areas.

Matches voronoi_area.m: U = 1 - std(areas, ddof=1) / mean(areas).
Unbounded cells and cells extending beyond the subplot are excluded, not clipped.
Coordinates must be in the same projected planar system (e.g. metres), not
longitude/latitude or uncorrected oblique-image pixels. All input hills enter
the tessellation, including any supplied neighbouring hills outside the subplot.

Input formats:
  --points: CSV with x,y in the first two columns; one header row by default.
  --boundary: ordered convex boundary vertices, x,y, without a header by default.
  With --plot-id, the boundary format is plot_id,x,y (as in the MATLAB script).
  Clockwise/counterclockwise vertices and a repeated closing vertex are accepted.

Example (run from the repository root):
  python spatial_uniformity.py --points hills.csv --boundary corners.csv
  python spatial_uniformity.py --points hills.csv --boundary plots.csv --plot-id 314

Required: NumPy and SciPy (already in requirements.txt).
Optional: --plot additionally requires matplotlib.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Voronoi
from scipy.spatial.qhull import QhullError


def calculate_uniformity(points, boundary):
    """Return summary, per-hill records and retained polygons in input units.

    A small numerical tolerance (1e-10 times the boundary span) treats vertices
    on an edge as inside, consistent with MATLAB inpolygon's boundary inclusion.
    The boundary must be convex; this includes rectangular field subplots.
    """
    points = np.asarray(points, dtype=float)
    boundary = np.asarray(boundary, dtype=float)
    for name, values, minimum in [('Points', points, 4), ('Boundary', boundary, 3)]:
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < minimum:
            raise ValueError('{} must contain at least {} x,y rows.'.format(name, minimum))
        if not np.isfinite(values).all():
            raise ValueError('{} contain non-finite coordinates.'.format(name))
    if np.array_equal(boundary[0], boundary[-1]):
        boundary = boundary[:-1]
    if len(boundary) < 3 or len(np.unique(boundary, axis=0)) != len(boundary):
        raise ValueError('Boundary requires at least three distinct vertices.')
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError('Duplicate hill coordinates must be resolved before tessellation.')

    # Translation improves numerical stability for large projected coordinates.
    origin = boundary.mean(axis=0)
    local_boundary = boundary - origin
    local_points = points - origin
    try:
        hull = ConvexHull(local_boundary)
    except QhullError as exc:
        raise ValueError('Boundary vertices must enclose a non-zero area.') from exc
    order = hull.vertices.tolist()
    n = len(boundary)
    forward = [(order[0] + i) % n for i in range(n)]
    backward = [(order[0] - i) % n for i in range(n)]
    if len(order) != n or (order != forward and order != backward):
        raise ValueError('Boundary must be convex, with vertices ordered around its perimeter.')

    # Hull equations give signed distances; non-positive values are inside.
    tolerance = 1e-10 * float(np.ptp(local_boundary, axis=0).max())
    try:
        voronoi = Voronoi(local_points)
    except QhullError as exc:
        raise ValueError('Cannot construct Voronoi cells; check for collinear or degenerate points.') from exc

    records, polygons, areas = [], [], []
    for index, region_index in enumerate(voronoi.point_region):
        region = voronoi.regions[region_index] if region_index >= 0 else []
        area, status = None, 'unbounded'
        if region and -1 not in region:
            vertices = voronoi.vertices[region]
            distances = vertices.dot(hull.equations[:, :2].T) + hull.equations[:, 2]
            if np.all(distances <= tolerance):
                # Shoelace area, with a local origin to avoid cancellation.
                shifted = vertices - vertices[0]
                x, y = shifted.T
                area = float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)
                if not np.isfinite(area) or area <= 0:
                    raise ValueError('Encountered a non-positive or invalid Voronoi area.')
                status = 'included'
                areas.append(area)
                polygons.append(vertices + origin)
            else:
                status = 'boundary_excluded'
        records.append((index + 1, float(points[index, 0]), float(points[index, 1]), area, status))

    if len(areas) < 2:
        raise ValueError('Fewer than two complete cells remain; uniformity is not reported.')
    mean_area = float(np.mean(areas))
    sample_std = float(np.std(areas, ddof=1))
    cv = sample_std / mean_area
    summary = {
        'input_hill_count': len(points),
        'valid_cell_count': len(areas),
        'unbounded_cell_count': sum(row[4] == 'unbounded' for row in records),
        'boundary_excluded_cell_count': sum(row[4] == 'boundary_excluded' for row in records),
        'mean_area': mean_area,
        'sample_std_area': sample_std,
        'coefficient_of_variation': cv,
        'spatial_uniformity': 1.0 - cv,
        'standard_deviation_ddof': 1,
    }
    # Do not clamp negative values: 1 - CV is not bounded below by zero.
    return summary, records, polygons


def load_table(path, delimiter, skiprows):
    """Read numeric CSV or whitespace-separated data with explicit header handling."""
    if skiprows < 0:
        raise ValueError('Header row counts must be non-negative.')
    return np.loadtxt(str(path), delimiter=None if delimiter == 'whitespace' else delimiter,
                      skiprows=skiprows, ndmin=2, encoding='utf-8-sig')


def save_plot(path, points, boundary, polygons, uniformity):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except ImportError as exc:
        raise ValueError('--plot requires matplotlib; install it or omit --plot.') from exc
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_collection(PolyCollection(polygons, facecolors='#ccefff', edgecolors='#aaaaaa',
                                     linewidths=0.6, alpha=0.7))
    closed = np.vstack([boundary, boundary[0]])
    ax.plot(closed[:, 0], closed[:, 1], color='#cc3333', linewidth=1.5)
    ax.scatter(points[:, 0], points[:, 1], s=12, c='#6db50a')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (input coordinate units)')
    ax.set_ylabel('Y (input coordinate units)')
    ax.set_title('Spatial uniformity = {:.6f}'.format(uniformity))
    fig.tight_layout()
    fig.savefig(str(path), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--points', required=True, type=Path, help='Numeric x,y hill coordinate file.')
    parser.add_argument('--boundary', required=True, type=Path, help='Subplot boundary coordinate file.')
    parser.add_argument('--plot-id', type=float, help='Select rows from a plot_id,x,y boundary table.')
    parser.add_argument('--points-skiprows', type=int, default=1, help='Point header rows (default: 1).')
    parser.add_argument('--boundary-skiprows', type=int, default=0, help='Boundary header rows (default: 0).')
    parser.add_argument('--delimiter', default=',', help='Point separator; use whitespace for TXT.')
    parser.add_argument('--boundary-delimiter', default=',', help='Boundary separator (default: comma).')
    parser.add_argument('--output-dir', type=Path, default=Path('uniformity_output'))
    parser.add_argument('--plot', action='store_true', help='Also save a PNG; requires matplotlib.')
    args = parser.parse_args()
    try:
        table = load_table(args.points, args.delimiter, args.points_skiprows)
        if table.shape[1] < 2:
            raise ValueError('Point file needs at least two columns: x,y.')
        points = table[:, :2]
        table = load_table(args.boundary, args.boundary_delimiter, args.boundary_skiprows)
        if args.plot_id is None:
            if table.shape[1] != 2:
                raise ValueError('Use --plot-id for a plot_id,x,y boundary table.')
            boundary = table
        else:
            if table.shape[1] != 3:
                raise ValueError('With --plot-id, boundary columns must be plot_id,x,y.')
            boundary = table[table[:, 0] == args.plot_id, 1:3]
        summary, records, polygons = calculate_uniformity(points, boundary)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.points.stem + ('_plot_{:g}'.format(args.plot_id) if args.plot_id is not None else '')
        if args.plot:
            save_plot(args.output_dir / (prefix + '_voronoi.png'), points, boundary,
                      polygons, summary['spatial_uniformity'])
        with (args.output_dir / (prefix + '_areas.csv')).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['hill_index', 'x', 'y', 'area', 'status'])
            writer.writerows(records)
        summary.update(points_file=str(args.points), boundary_file=str(args.boundary),
                       plot_id=args.plot_id, area_units='square input-coordinate units')
        with (args.output_dir / (prefix + '_summary.json')).open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write('\n')
        print('Spatial uniformity: {:.6f}'.format(summary['spatial_uniformity']))
        print('Valid cells: {} / {}'.format(summary['valid_cell_count'], summary['input_hill_count']))
        print('Results saved to: {}'.format(args.output_dir.resolve()))
    except (ValueError, OSError) as exc:
        parser.exit(1, 'Error: {}\n'.format(exc))


if __name__ == '__main__':
    main()
