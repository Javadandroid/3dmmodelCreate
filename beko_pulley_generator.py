#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


@dataclass(frozen=True)
class PulleyParams:
    teeth: int
    outer_diameter_mm: float
    tooth_depth_mm: float
    tooth_tip_fraction: float
    tooth_valley_fraction: float
    tooth_width_mm: float
    flange_diameter_mm: float
    flange_thickness_mm: float
    top_flange_thickness_mm: float
    bore_diameter_mm: float
    magnet_count: int
    magnet_diameter_mm: float
    magnet_depth_mm: float
    magnet_boss_diameter_mm: float
    magnet_boss_height_mm: float
    magnet_center_radius_mm: float
    magnet_angle_offset_deg: float
    magnet_boss_overlap_mm: float
    magnet_circle_segments: int


def _polar_xy(radius_mm: float, angle_rad: float) -> tuple[float, float]:
    return (radius_mm * math.cos(angle_rad), radius_mm * math.sin(angle_rad))


def _normalize_angles(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for angle_rad, radius_mm in points:
        angle_rad = angle_rad % (2.0 * math.pi)
        normalized.append((angle_rad, radius_mm))
    normalized.sort(key=lambda pair: pair[0])
    return normalized


def _make_tooth_profile(params: PulleyParams) -> list[tuple[float, float]]:
    if params.teeth < 6:
        raise ValueError("Number of teeth must be at least 6.")
    if params.outer_diameter_mm <= 0:
        raise ValueError("outer_diameter_mm must be > 0.")
    if not (0.05 <= params.tooth_tip_fraction <= 0.9):
        raise ValueError("tooth_tip_fraction must be between 0.05 and 0.9.")
    if not (0.0 <= params.tooth_valley_fraction <= 0.9):
        raise ValueError("tooth_valley_fraction must be between 0.0 and 0.9.")

    pitch_angle = 2.0 * math.pi / params.teeth
    tip_angle = pitch_angle * params.tooth_tip_fraction
    valley_angle = pitch_angle * params.tooth_valley_fraction
    flank_angle = (pitch_angle - tip_angle - valley_angle) / 2.0
    if flank_angle <= 0:
        raise ValueError(
            "Invalid fraction combination: tooth_tip_fraction + tooth_valley_fraction must be less than 1."
        )

    outer_radius = params.outer_diameter_mm / 2.0
    root_radius = outer_radius - params.tooth_depth_mm
    if root_radius <= 0:
        raise ValueError("tooth_depth_mm is too large (root radius <= 0).")

    points: list[tuple[float, float]] = []
    for tooth_index in range(params.teeth):
        start_angle = tooth_index * pitch_angle
        valley_end = start_angle + valley_angle
        tip_start = valley_end + flank_angle
        tip_end = tip_start + tip_angle
        next_start = start_angle + pitch_angle

        if tooth_index == 0:
            points.append((start_angle, root_radius))
        points.append((valley_end, root_radius))
        points.append((tip_start, outer_radius))
        points.append((tip_end, outer_radius))
        if tooth_index != params.teeth - 1:
            points.append((next_start, root_radius))

    return _normalize_angles(points)


def _make_constant_profile(angles: list[float], radius_mm: float) -> list[tuple[float, float]]:
    return [(angle_rad, radius_mm) for angle_rad in angles]


def _ring_vertices(profile: list[tuple[float, float]], z_mm: float) -> list[tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    for angle_rad, radius_mm in profile:
        x_mm, y_mm = _polar_xy(radius_mm, angle_rad)
        vertices.append((x_mm, y_mm, z_mm))
    return vertices


def _iter_edges(count: int) -> Iterable[tuple[int, int]]:
    for index in range(count):
        yield index, (index + 1) % count


def _circle_ring_vertices(
    radius_mm: float,
    z_mm: float,
    segments: int,
    center_xy: tuple[float, float] = (0.0, 0.0),
    angle_offset_rad: float = 0.0,
) -> list[tuple[float, float, float]]:
    cx_mm, cy_mm = center_xy
    vertices: list[tuple[float, float, float]] = []
    for index in range(segments):
        angle_rad = angle_offset_rad + 2.0 * math.pi * index / segments
        x_mm, y_mm = _polar_xy(radius_mm, angle_rad)
        vertices.append((cx_mm + x_mm, cy_mm + y_mm, z_mm))
    return vertices


def _add_disk_face(
    triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    center: tuple[float, float, float],
    ring: list[tuple[float, float, float]],
    reverse_winding: bool,
) -> None:
    count = len(ring)
    for index, next_index in _iter_edges(count):
        v1 = ring[index]
        v2 = ring[next_index]
        if reverse_winding:
            triangles.append((center, v2, v1))
        else:
            triangles.append((center, v1, v2))


def _stl_write_triangle(
    handle: TextIO, v1: tuple[float, float, float], v2: tuple[float, float, float], v3: tuple[float, float, float]
) -> None:
    handle.write("  facet normal 0 0 0\n")
    handle.write("    outer loop\n")
    handle.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
    handle.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
    handle.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
    handle.write("    endloop\n")
    handle.write("  endfacet\n")


def _add_side_surface(
    triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    bottom: list[tuple[float, float, float]],
    top: list[tuple[float, float, float]],
    reverse_winding: bool,
) -> None:
    if len(bottom) != len(top):
        raise ValueError("profile vertex counts must match")
    count = len(bottom)
    for index, next_index in _iter_edges(count):
        v00 = bottom[index]
        v01 = bottom[next_index]
        v11 = top[next_index]
        v10 = top[index]
        if reverse_winding:
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))
        else:
            triangles.append((v00, v11, v10))
            triangles.append((v00, v01, v11))


def _add_ring_face(
    triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    outer: list[tuple[float, float, float]],
    inner: list[tuple[float, float, float]],
    reverse_winding: bool,
) -> None:
    if len(outer) != len(inner):
        raise ValueError("ring vertex counts must match")
    count = len(outer)
    for index, next_index in _iter_edges(count):
        o0 = outer[index]
        o1 = outer[next_index]
        i1 = inner[next_index]
        i0 = inner[index]
        if reverse_winding:
            triangles.append((o0, i1, o1))
            triangles.append((o0, i0, i1))
        else:
            triangles.append((o0, o1, i1))
            triangles.append((o0, i1, i0))


def _add_magnet_boss_with_pocket(
    triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    center_xy: tuple[float, float],
    boss_outer_radius_mm: float,
    boss_z_bottom_mm: float,
    boss_z_top_mm: float,
    pocket_radius_mm: float,
    pocket_depth_mm: float,
    segments: int,
) -> None:
    if boss_outer_radius_mm <= 0:
        return
    if segments < 12:
        raise ValueError("magnet_circle_segments must be at least 12.")
    if pocket_radius_mm < 0:
        raise ValueError("magnet_diameter_mm must be >= 0.")
    if pocket_depth_mm < 0:
        raise ValueError("magnet_depth_mm must be >= 0.")
    if pocket_radius_mm >= boss_outer_radius_mm - 1e-6:
        raise ValueError("magnet_boss_diameter_mm must be larger than magnet_diameter_mm.")

    outer_bottom = _circle_ring_vertices(boss_outer_radius_mm, boss_z_bottom_mm, segments, center_xy=center_xy)
    outer_top = _circle_ring_vertices(boss_outer_radius_mm, boss_z_top_mm, segments, center_xy=center_xy)

    _add_side_surface(triangles, outer_bottom, outer_top, reverse_winding=False)
    _add_disk_face(triangles, (center_xy[0], center_xy[1], boss_z_top_mm), outer_top, reverse_winding=False)

    if pocket_radius_mm <= 0 or pocket_depth_mm <= 0:
        _add_disk_face(triangles, (center_xy[0], center_xy[1], boss_z_bottom_mm), outer_bottom, reverse_winding=True)
        return

    pocket_z_top_mm = boss_z_bottom_mm + pocket_depth_mm
    if pocket_z_top_mm >= boss_z_top_mm - 1e-6:
        raise ValueError("magnet_depth_mm must be less than the boss height for a blind pocket.")

    inner_bottom = _circle_ring_vertices(pocket_radius_mm, boss_z_bottom_mm, segments, center_xy=center_xy)
    inner_top = _circle_ring_vertices(pocket_radius_mm, pocket_z_top_mm, segments, center_xy=center_xy)

    _add_ring_face(triangles, outer_bottom, inner_bottom, reverse_winding=True)
    _add_side_surface(triangles, inner_bottom, inner_top, reverse_winding=True)
    _add_disk_face(triangles, (center_xy[0], center_xy[1], pocket_z_top_mm), inner_top, reverse_winding=True)


def generate_pulley_stl(params: PulleyParams, output_path: Path) -> None:
    tooth_profile = _make_tooth_profile(params)
    angles = [angle_rad for angle_rad, _ in tooth_profile]

    outer_radius = params.outer_diameter_mm / 2.0
    flange_radius = params.flange_diameter_mm / 2.0
    bore_radius = params.bore_diameter_mm / 2.0

    if flange_radius + 1e-6 < outer_radius:
        raise ValueError("For this simple model: flange_diameter_mm must be >= outer_diameter_mm.")
    if bore_radius <= 0:
        raise ValueError("bore_diameter_mm must be > 0.")
    if bore_radius + 1e-6 >= (outer_radius - params.tooth_depth_mm):
        raise ValueError("bore is too large and would intersect the tooth roots.")
    if params.tooth_width_mm <= 0:
        raise ValueError("tooth_width_mm must be > 0.")
    if params.flange_thickness_mm < 0:
        raise ValueError("flange_thickness_mm must be >= 0.")
    if params.top_flange_thickness_mm < 0:
        raise ValueError("top_flange_thickness_mm must be >= 0.")
    if params.magnet_count < 0:
        raise ValueError("magnet_count must be >= 0.")
    if params.magnet_count not in (0, 1, 2, 3, 4):
        raise ValueError("Currently only magnet_count values from 0 to 4 are supported.")
    if params.magnet_count:
        if params.magnet_diameter_mm <= 0:
            raise ValueError("magnet_diameter_mm must be > 0.")
        if params.magnet_depth_mm <= 0:
            raise ValueError("magnet_depth_mm must be > 0.")
        if params.magnet_boss_diameter_mm <= 0:
            raise ValueError("magnet_boss_diameter_mm must be > 0.")
        if params.magnet_boss_height_mm <= 0:
            raise ValueError("magnet_boss_height_mm must be > 0.")
        if params.magnet_center_radius_mm <= 0:
            raise ValueError("magnet_center_radius_mm must be > 0.")
        if params.magnet_boss_overlap_mm < 0:
            raise ValueError("magnet_boss_overlap_mm must be >= 0.")

    flange_profile = _make_constant_profile(angles, flange_radius)
    bore_profile = _make_constant_profile(angles, bore_radius)

    z0 = 0.0
    z1 = params.flange_thickness_mm
    z2 = z1 + params.tooth_width_mm
    z3 = z2 + params.top_flange_thickness_mm

    triangles: list[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    ] = []

    flange_bottom_outer = _ring_vertices(flange_profile, z0)
    flange_top_outer = _ring_vertices(flange_profile, z1)
    tooth_bottom_outer = _ring_vertices(tooth_profile, z1)
    tooth_top_outer = _ring_vertices(tooth_profile, z2)
    flange_bottom2_outer = _ring_vertices(flange_profile, z2)
    flange_top2_outer = _ring_vertices(flange_profile, z3)

    bore_bottom = _ring_vertices(bore_profile, z0)
    bore_top = _ring_vertices(bore_profile, z3)

    if params.flange_thickness_mm > 0:
        _add_side_surface(triangles, flange_bottom_outer, flange_top_outer, reverse_winding=False)
    if params.top_flange_thickness_mm > 0:
        _add_side_surface(triangles, flange_bottom2_outer, flange_top2_outer, reverse_winding=False)

    _add_side_surface(triangles, tooth_bottom_outer, tooth_top_outer, reverse_winding=False)

    _add_side_surface(triangles, bore_bottom, bore_top, reverse_winding=True)

    _add_ring_face(triangles, flange_bottom_outer, bore_bottom, reverse_winding=True)
    if params.top_flange_thickness_mm > 0:
        _add_ring_face(triangles, flange_top2_outer, bore_top, reverse_winding=False)
    else:
        _add_ring_face(triangles, tooth_top_outer, bore_top, reverse_winding=False)

    if flange_radius > outer_radius + 1e-6 and params.flange_thickness_mm > 0:
        _add_ring_face(triangles, flange_top_outer, tooth_bottom_outer, reverse_winding=False)
        if params.top_flange_thickness_mm > 0:
            _add_ring_face(triangles, flange_bottom2_outer, tooth_top_outer, reverse_winding=True)

    if params.magnet_count:
        boss_outer_radius = params.magnet_boss_diameter_mm / 2.0
        pocket_radius = params.magnet_diameter_mm / 2.0
        boss_z_bottom = z0 - params.magnet_boss_height_mm
        boss_z_top = z0 + params.magnet_boss_overlap_mm
        if boss_z_top <= boss_z_bottom:
            raise ValueError("magnet_boss_height_mm is invalid.")
        if params.magnet_depth_mm >= (boss_z_top - boss_z_bottom) - 1e-6:
            raise ValueError("magnet_depth_mm must be less than the boss height.")
        if params.magnet_center_radius_mm + boss_outer_radius >= flange_radius - 1e-6:
            raise ValueError("magnet_center_radius_mm is too large and the boss would extend past the flange.")
        if params.magnet_center_radius_mm - boss_outer_radius <= bore_radius + 1e-6:
            raise ValueError("magnet_center_radius_mm is too small and the boss would reach the bore.")

        angle_offset = math.radians(params.magnet_angle_offset_deg)
        for index in range(params.magnet_count):
            angle = angle_offset + 2.0 * math.pi * index / params.magnet_count
            cx, cy = _polar_xy(params.magnet_center_radius_mm, angle)
            _add_magnet_boss_with_pocket(
                triangles,
                center_xy=(cx, cy),
                boss_outer_radius_mm=boss_outer_radius,
                boss_z_bottom_mm=boss_z_bottom,
                boss_z_top_mm=boss_z_top,
                pocket_radius_mm=pocket_radius,
                pocket_depth_mm=params.magnet_depth_mm,
                segments=params.magnet_circle_segments,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"solid beko_pulley\n")
        for v1, v2, v3 in triangles:
            _stl_write_triangle(handle, v1, v2, v3)
        handle.write("endsolid beko_pulley\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a simple straight-tooth pulley STL (mm units).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--teeth", type=int, default=18, help="Number of teeth")
    parser.add_argument("--outer-d", type=float, default=16.0, help="Outer diameter at tooth tips (mm)")
    parser.add_argument("--tooth-depth", type=float, default=0.9, help="Tooth depth from tip to root (mm)")
    parser.add_argument("--tip-frac", type=float, default=0.40, help="Fraction of each pitch occupied by the tooth tip (0..1)")
    parser.add_argument("--valley-frac", type=float, default=0.20, help="Fraction of each pitch occupied by the valley (0..1)")
    parser.add_argument("--tooth-width", type=float, default=11.5, help="Length of the toothed section (mm)")
    parser.add_argument("--flange-d", type=float, default=20.0, help="Flange diameter (mm)")
    parser.add_argument("--flange-t", type=float, default=1.2, help="Flange thickness (mm)")
    parser.add_argument("--top-flange-t", type=float, default=0.0, help="Top flange thickness (mm)")
    parser.add_argument("--bore-d", type=float, default=11.0, help="Bore diameter (mm)")
    parser.add_argument("--magnet-count", type=int, default=2, help="Number of magnets (0 to disable)")
    parser.add_argument("--magnet-d", type=float, default=3.0, help="Magnet diameter (mm)")
    parser.add_argument("--magnet-depth", type=float, default=2.2, help="Magnet pocket depth (mm)")
    parser.add_argument("--magnet-boss-d", type=float, default=3.2, help="Magnet boss diameter (mm)")
    parser.add_argument("--magnet-boss-h", type=float, default=2.1, help="Magnet boss height (mm)")
    parser.add_argument("--magnet-center-r", type=float, default=7.5, help="Radius from part center to magnet boss center (mm)")
    parser.add_argument("--magnet-angle", type=float, default=45.0, help="Starting angle for the first magnet (degrees)")
    parser.add_argument("--magnet-overlap", type=float, default=0.3, help="Boss overlap with the part for adhesion in the slicer (mm)")
    parser.add_argument("--magnet-segments", type=int, default=48, help="Circle quality for magnet/boss (segment count)")
    parser.add_argument("--out", type=Path, default=Path("beko_ceg7425b_pulley.stl"), help="Output STL path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = PulleyParams(
        teeth=args.teeth,
        outer_diameter_mm=args.outer_d,
        tooth_depth_mm=args.tooth_depth,
        tooth_tip_fraction=args.tip_frac,
        tooth_valley_fraction=args.valley_frac,
        tooth_width_mm=args.tooth_width,
        flange_diameter_mm=args.flange_d,
        flange_thickness_mm=args.flange_t,
        top_flange_thickness_mm=args.top_flange_t,
        bore_diameter_mm=args.bore_d,
        magnet_count=args.magnet_count,
        magnet_diameter_mm=args.magnet_d,
        magnet_depth_mm=args.magnet_depth,
        magnet_boss_diameter_mm=args.magnet_boss_d,
        magnet_boss_height_mm=args.magnet_boss_h,
        magnet_center_radius_mm=args.magnet_center_r,
        magnet_angle_offset_deg=args.magnet_angle,
        magnet_boss_overlap_mm=args.magnet_overlap,
        magnet_circle_segments=args.magnet_segments,
    )
    generate_pulley_stl(params, args.out)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
