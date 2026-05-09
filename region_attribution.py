#!/usr/bin/env python3
"""
Match a city's metadata point to the urban-polygon it sits inside.

Used by:
  - backfill_polygons.py (one-shot R2 patcher)
  - VPS_Builder/pipeline.py (writes the polygon at index-build time)
  - inference server (reads back from config.json — no shapely needed there)

Polygon sources, in lookup order:
  US/*: data/United_States_Urban_By_County.geojson — `UAC_NAME` + county
  AU/*: data/Australia.geojson — `UCL_NAME21` + state

Output stored on each city's config.json:
  region_polygon: {"type": "Polygon"|"MultiPolygon", "coordinates": [...]}
  region_name:    human-readable label (e.g. "Buffalo / Erie")
  region_source:  "US_Urban_By_County" | "Australia"
  region_bbox:    [min_lng, min_lat, max_lng, max_lat] for fast circle prefiltering
  region_polygon_simplified: low-poly version (Douglas-Peucker, ~0.001°)
                             for sending to the browser without bloating /api/locations
"""

import json
import os
from pathlib import Path

from shapely.geometry import shape, Point
from shapely.ops import unary_union

# Default lookup paths; callers can override.
DEFAULT_US_PATH = r"D:\GeoAxis\Shapefiles\United_States_Urban_By_County.geojson"
DEFAULT_AU_PATH = r"D:\GeoAxis\Shapefiles\Australia.geojson"

SIMPLIFY_TOLERANCE = 0.001  # ~110 m at the equator; small enough for urban shapes


def _load_features(geojson_path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feats = []
    for feat in data["features"]:
        try:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            feats.append((geom, feat["properties"]))
        except Exception:
            continue
    return feats


class RegionResolver:
    """Lazy-loaded region polygon index for US + AU."""

    def __init__(self, us_path=DEFAULT_US_PATH, au_path=DEFAULT_AU_PATH):
        self.us_path = us_path
        self.au_path = au_path
        self._us = None
        self._au = None

    def _us_feats(self):
        if self._us is None:
            self._us = _load_features(self.us_path) if os.path.exists(self.us_path) else []
        return self._us

    def _au_feats(self):
        if self._au is None:
            self._au = _load_features(self.au_path) if os.path.exists(self.au_path) else []
        return self._au

    def find(self, lat, lng):
        """Return the matching polygon entry for a (lat, lng) point, or None.

        Result: {
            "region_polygon": GeoJSON geometry dict,
            "region_polygon_simplified": GeoJSON geometry dict,
            "region_bbox": [minx, miny, maxx, maxy],
            "region_name": str,
            "region_source": str,
        }
        """
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            return None
        pt = Point(lng, lat)
        # If multiple polygons contain the point (e.g. nested admin shapes),
        # the smallest by area wins — that's the most specific match.
        candidates = []
        for geom, props in self._us_feats():
            if geom.contains(pt):
                candidates.append((geom, props, "US_Urban_By_County"))
        for geom, props in self._au_feats():
            if geom.contains(pt):
                candidates.append((geom, props, "Australia"))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0].area)
        geom, props, source = candidates[0]
        return _to_record(geom, props, source)

    def find_union(self, points):
        """For an index that spans multiple admin polygons (e.g. NYC = 5 boroughs),
        sample many points, collect all distinct containing polygons, and return
        a single record where region_polygon is the union of all of them.

        Falls back to single-polygon `find` if all points hit the same shape.
        Returns None if nothing matches anywhere.
        """
        seen = {}  # source/name -> (geom, props, source)
        for lat, lng in points:
            if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
                continue
            pt = Point(lng, lat)
            best = None
            for geom, props in self._us_feats():
                if geom.contains(pt):
                    if best is None or geom.area < best[0].area:
                        best = (geom, props, "US_Urban_By_County")
            for geom, props in self._au_feats():
                if geom.contains(pt):
                    if best is None or geom.area < best[0].area:
                        best = (geom, props, "Australia")
            if best:
                key = (best[2], _name_for(best[1], best[2]))
                if key not in seen:
                    seen[key] = best
        if not seen:
            return None
        if len(seen) == 1:
            geom, props, source = next(iter(seen.values()))
            return _to_record(geom, props, source)
        # Multiple distinct polygons — union them.
        from shapely.ops import unary_union
        geoms = [v[0] for v in seen.values()]
        union = unary_union(geoms)
        if not union.is_valid:
            union = union.buffer(0)
        names = sorted(_name_for(v[1], v[2]) for v in seen.values())
        # Pick the source of the first contributor for `region_source`; use a
        # combined display name so the user can see what was matched.
        first_source = next(iter(seen.values()))[2]
        simplified = union.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
        if simplified.is_empty:
            simplified = union
        minx, miny, maxx, maxy = union.bounds
        # NOTE: we used to also serialise `region_polygon` (the full,
        # un-simplified geometry) but the inference server only reads
        # `region_polygon_simplified`. The full polygon was ~30× larger
        # (Brisbane: 735 KB vs 46 KB) and was bloating every config.json
        # in R2 + GCP for nothing, so it's dropped.
        return {
            "region_polygon_simplified": simplified.__geo_interface__,
            "region_bbox": [minx, miny, maxx, maxy],
            "region_name": " + ".join(names),
            "region_source": first_source,
            "region_count": len(seen),
        }


def _name_for(props, source):
    if source == "US_Urban_By_County":
        return f"{props.get('UAC_NAME','?')} / {props.get('COUNTY_NAME','?')}, {props.get('STATE_NAME','?')}"
    if source == "Australia":
        return f"{props.get('UCL_NAME21','?')}, {props.get('STE_NAME21','?')}"
    return "?"


def _to_record(geom, props, source):
    simplified = geom.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
    if simplified.is_empty:
        simplified = geom

    if source == "US_Urban_By_County":
        name = f"{props.get('UAC_NAME','?')} / {props.get('COUNTY_NAME','?')}, {props.get('STATE_NAME','?')}"
    elif source == "Australia":
        name = f"{props.get('UCL_NAME21','?')}, {props.get('STE_NAME21','?')}"
    else:
        name = "?"

    minx, miny, maxx, maxy = geom.bounds
    # See the multi-polygon branch above — we only ship the simplified
    # geometry; the un-simplified `region_polygon` was unused and bloated
    # every config.json by 30×.
    return {
        "region_polygon_simplified": simplified.__geo_interface__,
        "region_bbox": [minx, miny, maxx, maxy],
        "region_name": name,
        "region_source": source,
    }


# Convenience for callers that don't want to manage a singleton themselves.
_singleton = None


def find_region(lat, lng, us_path=DEFAULT_US_PATH, au_path=DEFAULT_AU_PATH):
    global _singleton
    if _singleton is None:
        _singleton = RegionResolver(us_path, au_path)
    return _singleton.find(lat, lng)
