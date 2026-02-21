#!/usr/bin/env python3
"""
Convert PoNHy result CSVs (or input topography/gravity text files) to GeoJSON.

PoNHy uses X, Y, Z in **meters (UTM)**. This script converts to WGS84 lon/lat
so you can view the data on web maps (e.g. geojson.io, QGIS, Mapbox).

Usage:
  python export_to_geojson.py path/to/Density_complete_model.csv -o out.geojson
  python export_to_geojson.py path/to/Ext_Grav.txt -o gravity.geojson --utm 30
  python export_to_geojson.py path/to/Ext_Topo.txt -o topo.geojson --format topo

Defaults: UTM zone 30N (Pyrenees). For California use e.g. --utm 11.
Requires: pyproj (pip install pyproj).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    try:
        import numpy as np
        from pyproj import Transformer
    except ImportError as e:
        print("Error: this script needs numpy and pyproj. Install with: pip install pyproj", file=sys.stderr)
        raise SystemExit(1) from e

    parser = argparse.ArgumentParser(
        description="Convert PoNHy X,Y,Z CSV/text to GeoJSON (WGS84)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to CSV or tab-separated file (X, Y, Z, ...). Default: latest Density_complete_model.csv in Results_Inversion_*.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output GeoJSON path. Default: <input_stem>.geojson next to input.",
    )
    parser.add_argument(
        "--utm",
        type=int,
        default=30,
        help="UTM zone number (e.g. 30 for Pyrenees, 11 for California). Default: 30.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "topo"],
        default="csv",
        help="Input format: 'csv' (header X,Y,Z,...) or 'topo' (no header, tab-separated X Y Z). Default: csv.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Max points to export (subsample if larger). Default: 50000.",
    )
    parser.add_argument(
        "--polygons",
        action="store_true",
        help="Export grid cells as polygons (rectangles) with values in properties, for coloring by value on the map.",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=None,
        help="Cell width in meters (for --polygons). Default: infer from grid spacing.",
    )
    parser.add_argument(
        "--dy",
        type=float,
        default=None,
        help="Cell height in meters (for --polygons). Default: infer from grid spacing.",
    )
    parser.add_argument(
        "--value-ranges",
        action="store_true",
        help="Merge grid cells into one polygon per value range (contour-style bands). Uses first value column (e.g. Density) or Z.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of value ranges for --value-ranges. Default: 10.",
    )
    args = parser.parse_args()

    if args.input is None:
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        for name in sorted(os.listdir(base)):
            if name.startswith("Results_Inversion_") and os.path.isdir(os.path.join(base, name)):
                path = os.path.join(base, name, "Density_complete_model.csv")
                if os.path.isfile(path):
                    candidates.append((path, os.path.getmtime(path)))
        if not candidates:
            print("No Results_Inversion_*/Density_complete_model.csv found. Pass an input file.", file=sys.stderr)
            raise SystemExit(1)
        candidates.sort(key=lambda x: x[1], reverse=True)
        args.input = candidates[0][0]
        print(f"Using latest: {args.input}", file=sys.stderr)

    if not os.path.isfile(args.input):
        print(f"File not found: {args.input}", file=sys.stderr)
        raise SystemExit(1)

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + ".geojson"

    # Load data
    if args.format == "csv":
        data = np.loadtxt(args.input, delimiter=",", skiprows=1, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x, y = data[:, 0], data[:, 1]
        z = data[:, 2] if data.shape[1] > 2 else np.full(len(x), np.nan)
        extra = data[:, 3:] if data.shape[1] > 3 else None
        header_row = open(args.input).readline().strip().split(",")
        extra_names = header_row[3:] if len(header_row) > 3 else [f"prop_{i}" for i in range(data.shape[1] - 3)]
    else:
        data = np.loadtxt(args.input, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x, y = data[:, 0], data[:, 1]
        z = data[:, 2] if data.shape[1] > 2 else np.full(len(x), np.nan)
        extra = data[:, 3:] if data.shape[1] > 3 else None
        extra_names = [f"col_{i}" for i in range(data.shape[1] - 3)] if extra is not None else []

    n = len(x)
    if n > args.max_points:
        # Subsample spatially so we don't get stripes (1D subsampling in file order looks goofy).
        ux = np.unique(x)
        uy = np.unique(y)
        nx, ny = len(ux), len(uy)
        if nx * ny >= n * 0.9 and nx > 1 and ny > 1:
            # Regular grid: keep every step_x-th column and step_y-th row
            target_side = max(1, int(args.max_points ** 0.5))
            step_x = max(1, nx // target_side)
            step_y = max(1, ny // target_side)
            ix = np.searchsorted(ux, x, side="right") - 1
            iy = np.searchsorted(uy, y, side="right") - 1
            ix = np.clip(ix, 0, nx - 1)
            iy = np.clip(iy, 0, ny - 1)
            keep = ((ix % step_x) == 0) & ((iy % step_y) == 0)
            idx = np.where(keep)[0]
            if len(idx) > args.max_points:
                idx = idx[np.linspace(0, len(idx) - 1, args.max_points, dtype=int)]
        else:
            idx = np.linspace(0, n - 1, args.max_points, dtype=int)
        x, y, z = x[idx], y[idx], z[idx]
        if extra is not None:
            extra = extra[idx]
        n = len(x)
        print(f"Subsampled to {n} points (--max-points {args.max_points}).", file=sys.stderr)

    # UTM (zone N) -> WGS84
    transformer = Transformer.from_crs(
        f"EPSG:326{args.utm:02d}",  # WGS 84 / UTM zone N
        "EPSG:4326",
        always_xy=True,
    )

    if args.value_ranges:
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            from shapely.ops import unary_union
            from shapely.ops import transform as shapely_transform
        except ImportError:
            print("Error: --value-ranges requires shapely. Install with: pip install shapely", file=sys.stderr)
            raise SystemExit(1)
        value = extra[:, 0] if (extra is not None and extra.shape[1] >= 1) else z
        value_name = extra_names[0] if (extra is not None and extra.shape[1] >= 1) else "z_m"
        n_bins = max(2, min(args.bins, 50))
        edges = np.percentile(value, np.linspace(0, 100, n_bins + 1))
        edges[0] = float(np.min(value))
        edges[-1] = float(np.max(value))
        bin_idx = np.searchsorted(edges, value, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        ux = np.unique(x)
        uy = np.unique(y)
        dx_infer = float(np.median(np.diff(np.sort(ux)))) if len(ux) > 1 else 500.0
        dy_infer = float(np.median(np.diff(np.sort(uy)))) if len(uy) > 1 else 500.0
        dx = args.dx if args.dx is not None else dx_infer
        dy = args.dy if args.dy is not None else dy_infer
        hx, hy = dx / 2.0, dy / 2.0
        def make_box(xc, yc):
            return ShapelyPolygon([
                (xc - hx, yc - hy), (xc + hx, yc - hy), (xc + hx, yc + hy), (xc - hx, yc + hy), (xc - hx, yc - hy),
            ])
        t_func = lambda x_pt, y_pt: transformer.transform(x_pt, y_pt)
        features = []
        for b in range(n_bins):
            mask = bin_idx == b
            if not np.any(mask):
                continue
            boxes = [make_box(x[i], y[i]) for i in np.where(mask)[0]]
            merged = unary_union(boxes)
            if merged.is_empty:
                continue
            merged_wgs = shapely_transform(t_func, merged)
            v_min, v_max = float(edges[b]), float(edges[b + 1])
            if hasattr(merged_wgs, "geoms"):
                geoms = merged_wgs.geoms
            else:
                geoms = [merged_wgs]
            for g in geoms:
                if g.is_empty or g.area < 1e-12:
                    continue
                if g.geom_type == "Polygon":
                    coords = [[list(c) for c in g.exterior.coords]]
                    for inner in g.interiors:
                        coords.append([list(c) for c in inner.coords])
                else:
                    coords = [[list(c) for c in g.exterior.coords]] if hasattr(g, "exterior") else []
                if not coords:
                    continue
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": coords},
                    "properties": {
                        "min_value": v_min,
                        "max_value": v_max,
                        "label": f"{v_min:.4g} - {v_max:.4g}",
                        "value_name": value_name,
                    },
                })
        fc = {"type": "FeatureCollection", "features": features}
        with open(args.output, "w") as f:
            json.dump(fc, f, separators=(",", ":"))
        print(f"Wrote {len(features)} polygon(s) for {n_bins} value ranges to {args.output}", file=sys.stderr)
    elif args.polygons:
        # Infer cell size from grid spacing if not given
        dx = args.dx
        dy = args.dy
        if dx is None or dy is None:
            ux = np.unique(x)
            uy = np.unique(y)
            if len(ux) > 1:
                dx_infer = float(np.median(np.diff(np.sort(ux))))
            else:
                dx_infer = 500.0
            if len(uy) > 1:
                dy_infer = float(np.median(np.diff(np.sort(uy))))
            else:
                dy_infer = 500.0
            if dx is None:
                dx = dx_infer
            if dy is None:
                dy = dy_infer
            print(f"Cell size: dx={dx:.1f} m, dy={dy:.1f} m", file=sys.stderr)

        features = []
        hx, hy = dx / 2.0, dy / 2.0
        for i in range(n):
            # Rectangle in UTM: (x-hx, y-hy), (x+hx, y-hy), (x+hx, y+hy), (x-hx, y+hy), close
            xc, yc = x[i], y[i]
            x_utm = np.array([xc - hx, xc + hx, xc + hx, xc - hx, xc - hx])
            y_utm = np.array([yc - hy, yc - hy, yc + hy, yc + hy, yc - hy])
            lon_ring, lat_ring = transformer.transform(x_utm, y_utm)
            coords = [[float(lon_ring[k]), float(lat_ring[k])] for k in range(5)]
            prop = {"z_m": float(z[i])}
            if extra is not None:
                for j, name in enumerate(extra_names):
                    prop[name] = float(extra[i, j])
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": prop,
            })
        fc = {"type": "FeatureCollection", "features": features}
        with open(args.output, "w") as f:
            json.dump(fc, f, separators=(",", ":"))
        print(f"Wrote {len(features)} polygon cells to {args.output}", file=sys.stderr)
    else:
        lon, lat = transformer.transform(x, y)
        features = []
        for i in range(n):
            prop = {"z_m": float(z[i])}
            if extra is not None:
                for j, name in enumerate(extra_names):
                    prop[name] = float(extra[i, j])
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon[i]), float(lat[i])]},
                "properties": prop,
            })
        fc = {"type": "FeatureCollection", "features": features}
        with open(args.output, "w") as f:
            json.dump(fc, f, separators=(",", ":"))
        print(f"Wrote {len(features)} points to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
