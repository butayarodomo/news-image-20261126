"""Generate the 1024x1024 museum-grade editorial world-news cover for Thursday, 26 November 2026."""

from __future__ import annotations

import math
from pathlib import Path

import cairo
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
FONTS_DIR = ASSETS / "fonts"
OUT_PATH = ROOT / "entry.png"

S = 2  # 2x supersampling (2048x2048 -> 1024x1024)
SIZE = 1024 * S

# Orthographic Earth framing (shows US East Coast -> Europe -> Africa -> Middle East -> China)
GLOBE_CX = 512.0
GLOBE_CY = 675.0
GLOBE_R = 475.0
LON0_DEG = 22.0
LAT0_DEG = 22.0

# Editorial Palette
MIDNIGHT_TOP = (4, 8, 20)
MIDNIGHT_LOW = (7, 16, 36)
OCEAN_DEEP = np.array([5.0, 14.0, 34.0], dtype=np.float32)
OCEAN_SHALLOW = np.array([15.0, 44.0, 88.0], dtype=np.float32)
LAND_NIGHT = np.array([15.0, 26.0, 46.0], dtype=np.float32)
LAND_TWILIGHT = np.array([34.0, 54.0, 82.0], dtype=np.float32)

AMBER = (245, 158, 11)
ORANGE = (255, 122, 24)
CRIMSON = (239, 68, 68)
CYAN = (34, 211, 238)
ICE_BLUE = (96, 165, 250)
IVORY = (248, 250, 252)
SLATE_TEXT = (203, 213, 225)


def project_lonlat(lon_deg: float, lat_deg: float) -> tuple[float, float, float]:
    """Forward orthographic spherical projection onto 1024-space canvas."""
    lon = math.radians(lon_deg - LON0_DEG)
    lat = math.radians(lat_deg)
    lat0 = math.radians(LAT0_DEG)
    cos_lat = math.cos(lat)
    sin_lat = math.sin(lat)
    cos_lat0 = math.cos(lat0)
    sin_lat0 = math.sin(lat0)

    x_s = cos_lat * math.sin(lon)
    y_s = -(cos_lat0 * sin_lat - sin_lat0 * cos_lat * math.cos(lon))
    z_s = sin_lat0 * sin_lat + cos_lat0 * cos_lat * math.cos(lon)
    return GLOBE_CX + GLOBE_R * x_s, GLOBE_CY + GLOBE_R * y_s, z_s


def inverse_project_globe(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray, in_globe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse orthographic projection from normalized sphere coords to (lon_deg, lat_deg)."""
    lon0 = math.radians(LON0_DEG)
    lat0 = math.radians(LAT0_DEG)
    cos_l0 = math.cos(lat0)
    sin_l0 = math.sin(lat0)

    # Note screen y points down, so sphere north y_up = -ny
    y_up = -ny
    rho = np.sqrt(nx * nx + y_up * y_up)
    c = np.arcsin(np.clip(rho, 0.0, 1.0))
    sin_c = np.sin(c)
    cos_c = nz

    safe_rho = np.where(rho > 1e-6, rho, 1.0)
    lat = np.where(
        in_globe,
        np.arcsin(np.clip(cos_c * sin_l0 + (y_up * sin_c * cos_l0) / safe_rho, -1.0, 1.0)),
        0.0,
    )
    lon = np.where(
        in_globe,
        lon0 + np.arctan2(nx * sin_c, safe_rho * cos_l0 * cos_c - y_up * sin_l0 * sin_c),
        0.0,
    )
    lon_deg = (np.degrees(lon) + 180.0) % 360.0 - 180.0
    lat_deg = np.degrees(lat)
    return lon_deg, lat_deg


def sample_equirectangular(tex: np.ndarray, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Bilinearly sample an equirectangular (H, W) texture at (lon_deg, lat_deg)."""
    h, w = tex.shape
    u = ((lon_deg + 180.0) / 360.0) * (w - 1)
    v = ((90.0 - lat_deg) / 180.0) * (h - 1)
    return map_coordinates(tex, [v, u], order=1, mode="nearest")


def render_base_globe() -> np.ndarray:
    """Render cosmos, stars, Rayleigh atmosphere, and Natural Earth 50m 3D globe."""
    rng = np.random.default_rng(20261126)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    dx = xx - GLOBE_CX * S
    dy = yy - GLOBE_CY * S
    r_px = np.sqrt(dx * dx + dy * dy)
    r_max = GLOBE_R * S
    in_globe = r_px <= r_max

    # 1. Deep midnight cosmos gradient + subtle diagonal Milky Way dust
    v_frac = yy / float(SIZE - 1)
    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    for c in range(3):
        canvas[:, :, c] = MIDNIGHT_TOP[c] * (1.0 - v_frac) + MIDNIGHT_LOW[c] * v_frac

    neb = np.exp(-((yy - 0.14 * SIZE + 0.16 * (xx - 0.5 * SIZE)) ** 2) / (2.0 * (180.0 * S) ** 2))
    canvas[:, :, 0] += neb * 5.0
    canvas[:, :, 1] += neb * 14.0
    canvas[:, :, 2] += neb * 28.0

    # Deterministic starfield in deep space around the globe
    n_stars = 520
    sx = rng.integers(0, SIZE, size=n_stars)
    sy = rng.integers(0, int(0.94 * SIZE), size=n_stars)
    sb = rng.uniform(0.22, 1.0, size=n_stars) ** 1.9
    sr = rng.choice([0, 1, 2], size=n_stars, p=[0.72, 0.21, 0.07])
    for i in range(n_stars):
        x0, y0 = int(sx[i]), int(sy[i])
        if math.hypot(x0 - GLOBE_CX * S, y0 - GLOBE_CY * S) <= r_max + 12 * S:
            continue
        bright = 210.0 * float(sb[i])
        tint = np.array([bright * 0.85, bright * 0.93, bright * 1.06], dtype=np.float32)
        if i % 6 == 0:
            tint = np.array([bright * 1.06, bright * 0.94, bright * 0.76], dtype=np.float32)
        rad = int(sr[i])
        y_lo, y_hi = max(0, y0 - rad), min(SIZE, y0 + rad + 1)
        x_lo, x_hi = max(0, x0 - rad), min(SIZE, x0 + rad + 1)
        for py in range(y_lo, y_hi):
            for px in range(x_lo, x_hi):
                fall = math.exp(-((px - x0) ** 2 + (py - y0) ** 2) / max(0.55, rad * 0.8))
                canvas[py, px] += tint * fall

    # 2. Outer Rayleigh atmospheric halo
    dist_out = np.maximum(0.0, r_px - r_max)
    atmo_outer = np.exp(-dist_out / (24.0 * S)) * (~in_globe)
    atmo_tight = np.exp(-dist_out / (4.5 * S)) * (~in_globe)
    sun_bias = np.clip(0.55 + 0.50 * ((dx / (r_max + 1.0)) * 0.65 - (dy / (r_max + 1.0)) * 0.65), 0.22, 1.32)

    canvas[:, :, 0] += (atmo_outer * 14.0 + atmo_tight * 88.0) * sun_bias
    canvas[:, :, 1] += (atmo_outer * 68.0 + atmo_tight * 178.0) * sun_bias
    canvas[:, :, 2] += (atmo_outer * 165.0 + atmo_tight * 246.0) * sun_bias

    # 3. Spherical geometry & Natural Earth 50m textures
    nx = np.where(in_globe, dx / r_max, 0.0)
    ny = np.where(in_globe, dy / r_max, 0.0)
    nz = np.where(in_globe, np.sqrt(np.maximum(0.0, 1.0 - nx * nx - ny * ny)), 0.0)

    lon_deg, lat_deg = inverse_project_globe(nx, ny, nz, in_globe)

    land_tex = np.asarray(Image.open(ASSETS / "earth_land_50m.png"), dtype=np.float32) / 255.0
    bord_tex = np.asarray(Image.open(ASSETS / "earth_borders_50m.png"), dtype=np.float32) / 255.0

    land = np.where(in_globe, sample_equirectangular(land_tex, lon_deg, lat_deg), 0.0)
    borders = np.where(in_globe, sample_equirectangular(bord_tex, lon_deg, lat_deg), 0.0)

    # Twilight / orbital sunlight vector from upper-east (gives dawn glow over Asia/Gulf, deep night over West)
    lx, ly, lz = 0.56, -0.45, 0.69
    ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    terminator = np.clip((ndotl - 0.12) / 0.72, 0.0, 1.0)
    fresnel = np.where(in_globe, (1.0 - nz) ** 3.1, 0.0)

    # Subtle procedural terrain/bathymetry relief
    relief = (
        0.5
        + 0.25 * np.sin(lon_deg * 0.22 + lat_deg * 0.17) * np.cos(lon_deg * 0.14 - lat_deg * 0.25)
        + 0.25 * np.sin(lon_deg * 0.51 - lat_deg * 0.43)
    )
    land_blur = gaussian_filter(land, sigma=1.4 * S)
    coast_rim = np.clip(land_blur * (1.0 - land) * 1.9, 0.0, 1.0)
    inland_depth = gaussian_filter(land, sigma=7.0 * S) * land

    ocean_col = OCEAN_DEEP[None, None, :] + (OCEAN_SHALLOW - OCEAN_DEEP)[None, None, :] * (terminator[:, :, None] ** 1.2)
    ocean_col += relief[:, :, None] * 2.2
    # Keep specular ocean highlight subtle & away from bottom footer
    hx, hy, hz = 0.34, -0.38, 0.86
    spec = (np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0) ** 22.0) * np.clip(1.0 - yy / (0.86 * SIZE), 0.0, 1.0)
    ocean_col += spec[:, :, None] * np.array([22.0, 54.0, 88.0], dtype=np.float32) * (1.0 - land[:, :, None])

    land_col = LAND_NIGHT[None, None, :] + (LAND_TWILIGHT - LAND_NIGHT)[None, None, :] * (terminator[:, :, None] ** 1.05)
    land_col += inland_depth[:, :, None] * np.array([7.0, 10.0, 14.0], dtype=np.float32)
    land_col += (relief[:, :, None] - 0.5) * 5.0

    globe_rgb = ocean_col * (1.0 - land[:, :, None]) + land_col * land[:, :, None]
    # Subtle cyan-steel coastline edge & faint national borders
    globe_rgb += coast_rim[:, :, None] * np.array([24.0, 74.0, 122.0], dtype=np.float32)
    globe_rgb += (borders * land)[:, :, None] * np.array([22.0, 38.0, 58.0], dtype=np.float32) * 0.55
    globe_rgb += fresnel[:, :, None] * np.array([26.0, 98.0, 188.0], dtype=np.float32) * sun_bias[:, :, None]

    # 4. Real Natural Earth 50m city lights (dimmed in Ukraine blackout corridor)
    cities = np.load(ASSETS / "earth_cities.npz")["cities"]
    light_layer = np.zeros((SIZE, SIZE), dtype=np.float32)
    for clon, clat, pop in cities:
        px, py, pz = project_lonlat(float(clon), float(clat))
        if pz <= 0.06:
            continue
        # Dim Ukrainian grid cities to reflect Nov 26 multi-region power outages
        in_ukraine = (23.0 <= clon <= 39.5) and (45.0 <= clat <= 52.2)
        strength = min(1.3, max(0.18, math.log10(max(pop, 1e4)) - 3.9))
        if in_ukraine:
            strength *= 0.14
        ox = int(px * S)
        oy = int(py * S)
        if 2 <= ox < SIZE - 2 and 2 <= oy < SIZE - 2:
            light_layer[oy - 1 : oy + 1, ox - 1 : ox + 1] += strength
        # Subtle metropolitan scatter for major hubs
        if pop >= 1_500_000 and not in_ukraine:
            for _ in range(5):
                jx = int((px + rng.normal(0, 3.2)) * S)
                jy = int((py + rng.normal(0, 2.6)) * S)
                if 2 <= jx < SIZE - 2 and 2 <= jy < SIZE - 2:
                    light_layer[jy, jx] += strength * 0.45

    light_soft = gaussian_filter(light_layer, sigma=1.35 * S)
    globe_rgb[:, :, 0] += light_soft * 84.0
    globe_rgb[:, :, 1] += light_soft * 60.0
    globe_rgb[:, :, 2] += light_soft * 22.0

    canvas[in_globe] = globe_rgb[in_globe]
    return np.clip(canvas, 0.0, 255.0)


def cairo_surface_to_rgba(surf: cairo.ImageSurface) -> np.ndarray:
    """Convert BGRA premultiplied Cairo surface to float RGBA [0..255] array."""
    raw = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape(SIZE, SIZE, 4)
    rgba = np.empty((SIZE, SIZE, 4), dtype=np.float32)
    rgba[:, :, 0] = raw[:, :, 2]
    rgba[:, :, 1] = raw[:, :, 1]
    rgba[:, :, 2] = raw[:, :, 0]
    rgba[:, :, 3] = raw[:, :, 3]
    alpha = rgba[:, :, 3:4] / 255.0
    mask = alpha > 1e-4
    rgba[:, :, :3] = np.where(mask, np.minimum(255.0, rgba[:, :, :3] / np.maximum(alpha, 1e-4)), 0.0)
    return rgba


def draw_great_arc(
    ctx: cairo.Context,
    lon0: float,
    lat0: float,
    lon1: float,
    lat1: float,
    lift_px: float,
    color: tuple[float, float, float, float],
    width: float,
    dashed: bool = False,
    steps: int = 36,
) -> None:
    """Draw a smooth geodesic-like elevated arc between two real world coordinates."""
    pts: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / float(steps)
        lon = lon0 + (lon1 - lon0) * t
        lat = lat0 + (lat1 - lat0) * t
        px, py, pz = project_lonlat(lon, lat)
        if pz < -0.05:
            continue
        arch = math.sin(math.pi * t) * lift_px
        pts.append((px * S, (py - arch) * S))
    if len(pts) < 2:
        return
    ctx.save()
    ctx.set_source_rgba(*color)
    ctx.set_line_width(width * S)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    if dashed:
        ctx.set_dash([8.0 * S, 6.0 * S])
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.stroke()
    ctx.restore()


def draw_curved_screen_arc(
    ctx: cairo.Context,
    p0: tuple[float, float],
    ctrl: tuple[float, float],
    p1: tuple[float, float],
    color: tuple[float, float, float, float],
    width: float,
    dashed: bool = False,
) -> None:
    ctx.save()
    ctx.set_source_rgba(*color)
    ctx.set_line_width(width * S)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    if dashed:
        ctx.set_dash([9.0 * S, 7.0 * S])
    ctx.move_to(p0[0] * S, p0[1] * S)
    ctx.curve_to(ctrl[0] * S, ctrl[1] * S, ctrl[0] * S, ctrl[1] * S, p1[0] * S, p1[1] * S)
    ctx.stroke()
    ctx.restore()


def draw_tiangong_station(ctx: cairo.Context, cx: float, cy: float, angle_deg: float = -12.0) -> None:
    """Detailed editorial Tiangong station + docked Shenzhou-24 with photovoltaic grids."""
    ctx.save()
    ctx.translate(cx * S, cy * S)
    ctx.rotate(math.radians(angle_deg))
    sc = 0.86 * S

    def solar_wing(x: float, y: float, w: float, h: float, tilt: float = 0.0) -> None:
        ctx.save()
        ctx.translate(x * sc, y * sc)
        ctx.rotate(math.radians(tilt))
        ctx.set_source_rgba(0.84, 0.94, 1.0, 0.96)
        ctx.set_line_width(2.0 * sc)
        ctx.move_to(0, -h * 0.54 * sc)
        ctx.line_to(0, h * 0.54 * sc)
        ctx.stroke()

        grad = cairo.LinearGradient(-w * 0.5 * sc, -h * 0.5 * sc, w * 0.5 * sc, h * 0.5 * sc)
        grad.add_color_stop_rgba(0.0, 0.16, 0.64, 0.92, 0.97)
        grad.add_color_stop_rgba(0.5, 0.07, 0.32, 0.58, 0.97)
        grad.add_color_stop_rgba(1.0, 0.28, 0.84, 1.0, 0.97)
        ctx.set_source(grad)
        ctx.rectangle(-w * 0.5 * sc, -h * 0.5 * sc, w * sc, h * sc)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.78, 0.97, 1.0, 0.90)
        ctx.set_line_width(1.1 * sc)
        ctx.stroke()

        ctx.set_source_rgba(0.58, 0.92, 1.0, 0.45)
        ctx.set_line_width(0.75 * sc)
        for i in range(1, 6):
            yy = (-0.5 + i / 6.0) * h * sc
            ctx.move_to(-w * 0.5 * sc, yy)
            ctx.line_to(w * 0.5 * sc, yy)
        for j in range(1, 3):
            xx = (-0.5 + j / 3.0) * w * sc
            ctx.move_to(xx, -h * 0.5 * sc)
            ctx.line_to(xx, h * 0.5 * sc)
        ctx.stroke()
        ctx.restore()

    solar_wing(-74, -48, 25, 68, -4)
    solar_wing(-74, 48, 25, 68, -4)
    solar_wing(74, -48, 25, 68, 4)
    solar_wing(74, 48, 25, 68, 4)
    solar_wing(-18, -34, 16, 36, -5)
    solar_wing(-18, 34, 16, 36, -5)

    arm_grad = cairo.LinearGradient(0, -9 * sc, 0, 9 * sc)
    arm_grad.add_color_stop_rgba(0.0, 0.99, 0.99, 1.0, 1.0)
    arm_grad.add_color_stop_rgba(0.55, 0.78, 0.87, 0.95, 1.0)
    arm_grad.add_color_stop_rgba(1.0, 0.42, 0.54, 0.68, 1.0)
    ctx.set_source(arm_grad)
    ctx.rectangle(-96 * sc, -6.5 * sc, 192 * sc, 13 * sc)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.45, 0.88, 1.0, 0.70)
    ctx.set_line_width(1.0 * sc)
    ctx.stroke()

    # Module seam bands
    ctx.set_source_rgba(0.18, 0.35, 0.52, 0.55)
    for mx in [-46, -22, 22, 46]:
        ctx.rectangle(mx * sc, -6.5 * sc, 2 * sc, 13 * sc)
    ctx.fill()

    core_grad = cairo.LinearGradient(-10 * sc, 0, 10 * sc, 0)
    core_grad.add_color_stop_rgba(0.0, 0.52, 0.64, 0.76, 1.0)
    core_grad.add_color_stop_rgba(0.4, 0.97, 0.99, 1.0, 1.0)
    core_grad.add_color_stop_rgba(1.0, 0.70, 0.82, 0.93, 1.0)
    ctx.set_source(core_grad)
    ctx.rectangle(-7.5 * sc, -44 * sc, 15 * sc, 60 * sc)
    ctx.fill()

    ctx.set_source_rgba(0.96, 0.99, 1.0, 1.0)
    ctx.arc(0, 0, 10.5 * sc, 0, 2 * math.pi)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.22, 0.86, 0.98, 0.95)
    ctx.set_line_width(1.3 * sc)
    ctx.stroke()

    # Docked Shenzhou-24 spacecraft
    ctx.set_source_rgba(0.93, 0.97, 1.0, 1.0)
    ctx.rectangle(-5.0 * sc, 12 * sc, 10.0 * sc, 24 * sc)
    ctx.fill()
    ctx.set_source_rgba(0.28, 0.88, 1.0, 0.96)
    ctx.rectangle(-21 * sc, 22 * sc, 14 * sc, 6 * sc)
    ctx.rectangle(7 * sc, 22 * sc, 14 * sc, 6 * sc)
    ctx.fill()
    ctx.set_source_rgba(0.99, 1.0, 1.0, 1.0)
    ctx.arc(0, 38 * sc, 5.2 * sc, 0, math.pi)
    ctx.fill()

    ctx.restore()


def draw_container_ship(ctx: cairo.Context, cx: float, cy: float) -> None:
    """Detailed container vessel with perspective stacked containers on Atlantic lane."""
    ctx.save()
    ctx.translate(cx * S, cy * S)
    sc = 0.78 * S

    # Wake
    ctx.set_source_rgba(0.96, 0.62, 0.04, 0.36)
    ctx.move_to(-82 * sc, 18 * sc)
    ctx.line_to(90 * sc, 18 * sc)
    ctx.line_to(104 * sc, 22 * sc)
    ctx.line_to(-96 * sc, 22 * sc)
    ctx.close_path()
    ctx.fill()

    # Steel hull with sunlit sheer
    ctx.set_source_rgba(0.25, 0.33, 0.45, 0.98)
    ctx.move_to(-76 * sc, 17 * sc)
    ctx.line_to(70 * sc, 17 * sc)
    ctx.line_to(86 * sc, 1 * sc)
    ctx.line_to(-80 * sc, 3 * sc)
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(0.72, 0.82, 0.94, 0.78)
    ctx.set_line_width(1.2 * sc)
    ctx.stroke()

    ctx.set_source_rgba(0.86, 0.22, 0.22, 0.90)
    ctx.rectangle(-76 * sc, 13.5 * sc, 148 * sc, 3.5 * sc)
    ctx.fill()

    # Stern bridge + funnel
    ctx.set_source_rgba(0.86, 0.91, 0.97, 0.98)
    ctx.rectangle(-68 * sc, -19 * sc, 16 * sc, 22 * sc)
    ctx.fill()
    ctx.set_source_rgba(0.96, 0.62, 0.04, 0.92)
    ctx.rectangle(-65 * sc, -15 * sc, 10 * sc, 3 * sc)
    ctx.fill()

    palette = [
        (0.92, 0.60, 0.08, 0.97),
        (0.38, 0.49, 0.63, 0.97),
        (0.86, 0.26, 0.23, 0.96),
        (0.50, 0.61, 0.75, 0.97),
    ]
    for col in range(7):
        tiers = 3 if col < 5 else 2
        for row in range(tiers):
            c = palette[(col + row * 2) % len(palette)]
            ctx.set_source_rgba(*c)
            x0 = (-44 + col * 16.0) * sc
            y0 = (-4.0 - row * 6.8) * sc
            ctx.rectangle(x0, y0, 14.0 * sc, 5.8 * sc)
            ctx.fill()

    ctx.restore()


def draw_tariff_chevrons(ctx: cairo.Context, x: float, y: float, angle_deg: float = 0.0) -> None:
    """Draw sharp amber/red tariff barrier chevrons intersecting a trade arc."""
    ctx.save()
    ctx.translate(x * S, y * S)
    ctx.rotate(math.radians(angle_deg))
    for i, col in enumerate([(0.96, 0.62, 0.04, 0.98), (0.94, 0.27, 0.27, 0.98)]):
        ox = (i - 0.5) * 9 * S
        ctx.set_source_rgba(*col)
        ctx.set_line_width(2.8 * S)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(ox - 4.5 * S, -9 * S)
        ctx.line_to(ox + 3.5 * S, 0)
        ctx.line_to(ox - 4.5 * S, 9 * S)
        ctx.stroke()
    ctx.restore()


def draw_ukraine_vignette(ctx: cairo.Context, cx: float, cy: float) -> None:
    """Damaged winter apartment block + fractured high-voltage pylon + outage sparks over Ukraine."""
    ctx.save()
    ctx.translate(cx * S, cy * S)
    sc = 0.78 * S

    # Cold winter fog halo
    fog = cairo.RadialGradient(0, 0, 8 * sc, 0, 0, 68 * sc)
    fog.add_color_stop_rgba(0.0, 0.07, 0.14, 0.28, 0.75)
    fog.add_color_stop_rgba(1.0, 0.04, 0.08, 0.18, 0.0)
    ctx.set_source(fog)
    ctx.arc(0, 0, 68 * sc, 0, 2 * math.pi)
    ctx.fill()

    # Darkened apartment silhouettes (brighter steel-slate edges + damaged roofline)
    ctx.set_source_rgba(0.24, 0.32, 0.44, 0.96)
    ctx.rectangle(-58 * sc, -4 * sc, 24 * sc, 30 * sc)
    ctx.rectangle(-30 * sc, -16 * sc, 22 * sc, 42 * sc)
    # Damaged notch on central building
    ctx.move_to(0 * sc, 26 * sc)
    ctx.line_to(0 * sc, -12 * sc)
    ctx.line_to(10 * sc, -12 * sc)
    ctx.line_to(14 * sc, -4 * sc)
    ctx.line_to(22 * sc, -8 * sc)
    ctx.line_to(22 * sc, 26 * sc)
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(0.55, 0.67, 0.80, 0.55)
    ctx.set_line_width(1.0 * sc)
    ctx.stroke()

    # Sparse unlit/amber windows on left apartment
    ctx.set_source_rgba(0.96, 0.62, 0.12, 0.55)
    ctx.rectangle(-52 * sc, 4 * sc, 4 * sc, 4 * sc)
    ctx.rectangle(-24 * sc, -8 * sc, 4 * sc, 4 * sc)
    ctx.fill()

    # Fractured steel power pylon on right
    px, py, h = 44.0, 26.0, 66.0
    ctx.save()
    ctx.translate(px * sc, py * sc)
    ctx.set_source_rgba(0.70, 0.80, 0.92, 0.96)
    ctx.set_line_width(1.6 * sc)
    wb, wt = h * 0.27, h * 0.08
    ctx.move_to(-wb, 0)
    ctx.line_to(-wt, -h * 0.88)
    ctx.line_to(0, -h)
    ctx.line_to(wt, -h * 0.88)
    ctx.line_to(wb, 0)
    ctx.stroke()
    for frac, span in [(0.52, 0.34), (0.76, 0.25)]:
        yy = -h * frac
        ctx.move_to(-h * span, yy)
        ctx.line_to(h * span, yy)
        ctx.stroke()
    ctx.restore()

    # Severed line & electric outage arc
    ctx.set_source_rgba(0.94, 0.27, 0.27, 0.94)
    ctx.set_line_width(1.6 * sc)
    ctx.move_to(22 * sc, -24 * sc)
    ctx.curve_to(6 * sc, -10 * sc, 4 * sc, 2 * sc, 10 * sc, 10 * sc)
    ctx.stroke()

    for sx, sy, col in [(10, 10, (0.98, 0.65, 0.12)), (14, -6, (0.94, 0.27, 0.27))]:
        ctx.set_source_rgba(*col, 0.98)
        ctx.arc(sx * sc, sy * sc, 3.4 * sc, 0, 2 * math.pi)
        ctx.fill()
        ctx.set_line_width(1.3 * sc)
        for ang in range(0, 360, 60):
            r = math.radians(ang)
            ctx.move_to(sx * sc, sy * sc)
            ctx.line_to((sx + 9 * math.cos(r)) * sc, (sy + 9 * math.sin(r)) * sc)
        ctx.stroke()

    # Incoming drone/missile tracks in winter sky
    for (x0, y0, x1, y1) in [(-56, -48, -14, -22), (62, -52, 24, -20)]:
        g = cairo.LinearGradient(x0 * sc, y0 * sc, x1 * sc, y1 * sc)
        g.add_color_stop_rgba(0.0, 0.94, 0.27, 0.27, 0.0)
        g.add_color_stop_rgba(1.0, 0.98, 0.45, 0.30, 0.90)
        ctx.set_source(g)
        ctx.set_line_width(1.8 * sc)
        ctx.move_to(x0 * sc, y0 * sc)
        ctx.line_to(x1 * sc, y1 * sc)
        ctx.stroke()

    ctx.restore()


def draw_diplomacy_carrier(ctx: cairo.Context, cx: float, cy: float) -> None:
    """Crisp Nimitz-style aircraft carrier & tactical radar arcs."""
    ctx.save()
    ctx.translate(cx * S, cy * S)
    sc = 0.78 * S

    for rad, al in [(26, 0.55), (44, 0.36), (62, 0.22)]:
        ctx.set_source_rgba(0.38, 0.65, 0.98, al)
        ctx.set_line_width(1.3 * sc)
        ctx.arc(0, -6 * sc, rad * sc, math.radians(-160), math.radians(-20))
        ctx.stroke()

    grad = cairo.RadialGradient(0, -6 * sc, 4 * sc, 0, -6 * sc, 62 * sc)
    grad.add_color_stop_rgba(0.0, 0.38, 0.65, 0.98, 0.30)
    grad.add_color_stop_rgba(1.0, 0.38, 0.65, 0.98, 0.0)
    ctx.set_source(grad)
    ctx.move_to(0, -6 * sc)
    ctx.arc(0, -6 * sc, 62 * sc, math.radians(-115), math.radians(-60))
    ctx.close_path()
    ctx.fill()

    ctx.set_source_rgba(0.38, 0.65, 0.98, 0.24)
    ctx.rectangle(-90 * sc, 13 * sc, 180 * sc, 3 * sc)
    ctx.fill()

    ctx.set_source_rgba(0.36, 0.45, 0.58, 0.98)
    ctx.move_to(-80 * sc, 13 * sc)
    ctx.line_to(72 * sc, 13 * sc)
    ctx.line_to(88 * sc, 1 * sc)
    ctx.line_to(-84 * sc, 1 * sc)
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(0.75, 0.85, 0.96, 0.85)
    ctx.set_line_width(1.2 * sc)
    ctx.stroke()

    ctx.set_source_rgba(0.55, 0.65, 0.78, 0.98)
    ctx.rectangle(16 * sc, -17 * sc, 20 * sc, 18 * sc)
    ctx.fill()
    ctx.set_line_width(1.6 * sc)
    ctx.move_to(26 * sc, -17 * sc)
    ctx.line_to(26 * sc, -28 * sc)
    ctx.stroke()

    ctx.set_source_rgba(0.32, 0.40, 0.52, 0.95)
    ctx.move_to(-120 * sc, 16 * sc)
    ctx.line_to(-90 * sc, 16 * sc)
    ctx.line_to(-86 * sc, 10 * sc)
    ctx.line_to(-118 * sc, 10 * sc)
    ctx.close_path()
    ctx.fill()
    ctx.rectangle(-108 * sc, 3 * sc, 9 * sc, 7 * sc)
    ctx.fill()

    ctx.restore()


def draw_cairo_viaduct(ctx: cairo.Context, cx: float, cy: float) -> None:
    """Multi-span Cairo Ring Road elevated overpass with emergency beacons & controlled tanker fire."""
    ctx.save()
    ctx.translate(cx * S, cy * S)
    sc = 0.78 * S

    ctx.set_source_rgba(0.04, 0.07, 0.13, 0.85)
    ctx.rectangle(-92 * sc, 23 * sc, 184 * sc, 8 * sc)
    ctx.fill()

    ctx.set_source_rgba(0.28, 0.35, 0.46, 0.96)
    for px in [-66, -22, 22, 66]:
        ctx.move_to((px - 5) * sc, 26 * sc)
        ctx.line_to((px - 3.5) * sc, 6 * sc)
        ctx.line_to((px + 3.5) * sc, 6 * sc)
        ctx.line_to((px + 5) * sc, 26 * sc)
        ctx.close_path()
        ctx.fill()

    ctx.set_source_rgba(0.38, 0.47, 0.60, 0.98)
    ctx.move_to(-94 * sc, 8 * sc)
    ctx.curve_to(-30 * sc, 0 * sc, 30 * sc, 0 * sc, 94 * sc, 8 * sc)
    ctx.line_to(94 * sc, 14 * sc)
    ctx.curve_to(30 * sc, 6 * sc, -30 * sc, 6 * sc, -94 * sc, 14 * sc)
    ctx.close_path()
    ctx.fill_preserve()
    ctx.set_source_rgba(0.96, 0.62, 0.14, 0.80)
    ctx.set_line_width(1.2 * sc)
    ctx.stroke()

    for bx, col in [(-52, (0.94, 0.27, 0.27)), (-40, (0.38, 0.65, 0.98)), (44, (0.94, 0.27, 0.27)), (56, (0.38, 0.65, 0.98))]:
        ctx.set_source_rgba(*col, 0.98)
        ctx.arc(bx * sc, 1.5 * sc, 2.5 * sc, 0, 2 * math.pi)
        ctx.fill()

    ctx.set_source_rgba(0.14, 0.16, 0.22, 0.97)
    ctx.rectangle(-15 * sc, -4 * sc, 25 * sc, 8 * sc)
    ctx.fill()

    for sx, sy, sr, al in [(-2, -16, 12, 0.62), (8, -28, 16, 0.52), (20, -40, 20, 0.36), (34, -50, 22, 0.22)]:
        ctx.set_source_rgba(0.10, 0.12, 0.17, al)
        ctx.arc(sx * sc, sy * sc, sr * sc, 0, 2 * math.pi)
        ctx.fill()

    fire_g = cairo.RadialGradient(-2 * sc, -4 * sc, 1 * sc, -2 * sc, -6 * sc, 19 * sc)
    fire_g.add_color_stop_rgba(0.0, 1.0, 0.93, 0.65, 0.98)
    fire_g.add_color_stop_rgba(0.45, 1.0, 0.48, 0.10, 0.92)
    fire_g.add_color_stop_rgba(1.0, 0.94, 0.27, 0.12, 0.0)
    ctx.set_source(fire_g)
    ctx.arc(-2 * sc, -6 * sc, 19 * sc, 0, 2 * math.pi)
    ctx.fill()

    ctx.restore()


def draw_ras_tanura_refinery(ctx: cairo.Context, cx: float, cy: float) -> None:
    """Ras Tanura refinery towers/tanks with satellite-style smoke plume & Red Sea defense arcs."""
    ctx.save()
    ctx.translate(cx * S, cy * S)
    sc = 0.78 * S

    for (x0, y0, cx0, cy0, x1, y1, col) in [
        (-74, -22, -32, -74, 12, -52, (0.94, 0.27, 0.27, 0.92)),
        (64, -18, 32, -72, 12, -52, (0.90, 0.96, 1.0, 0.94)),
    ]:
        ctx.set_source_rgba(*col)
        ctx.set_line_width(1.9 * sc)
        ctx.set_dash([5.0 * sc, 4.0 * sc])
        ctx.move_to(x0 * sc, y0 * sc)
        ctx.curve_to(cx0 * sc, cy0 * sc, cx0 * sc, cy0 * sc, x1 * sc, y1 * sc)
        ctx.stroke()
    ctx.set_dash([])

    ctx.set_source_rgba(1.0, 0.88, 0.55, 0.98)
    ctx.arc(12 * sc, -52 * sc, 4.0 * sc, 0, 2 * math.pi)
    ctx.fill()

    ctx.set_source_rgba(0.32, 0.40, 0.52, 0.97)
    for tx, tw, th in [(-42, 8, 44), (-28, 10, 54), (-12, 7, 38), (4, 6, 48)]:
        ctx.rectangle(tx * sc, (24 - th) * sc, tw * sc, th * sc)
    ctx.fill()

    ctx.set_source_rgba(0.42, 0.50, 0.63, 0.97)
    for kx in [22, 40, 58]:
        ctx.rectangle(kx * sc, 8 * sc, 14 * sc, 16 * sc)
        ctx.arc((kx + 7) * sc, 8 * sc, 7 * sc, math.pi, 2 * math.pi)
    ctx.fill()

    ctx.set_source_rgba(0.22, 0.28, 0.39, 0.98)
    ctx.rectangle(-68 * sc, 22 * sc, 146 * sc, 7 * sc)
    ctx.fill()

    for sx, sy, sr, al in [
        (-22, -26, 13, 0.70),
        (-36, -40, 17, 0.62),
        (-54, -54, 22, 0.48),
        (-74, -66, 26, 0.34),
    ]:
        ctx.set_source_rgba(0.08, 0.10, 0.14, al)
        ctx.arc(sx * sc, sy * sc, sr * sc, 0, 2 * math.pi)
        ctx.fill()

    fg = cairo.RadialGradient(-22 * sc, -18 * sc, 2 * sc, -22 * sc, -18 * sc, 21 * sc)
    fg.add_color_stop_rgba(0.0, 1.0, 0.85, 0.45, 0.97)
    fg.add_color_stop_rgba(0.5, 0.96, 0.45, 0.08, 0.84)
    fg.add_color_stop_rgba(1.0, 0.94, 0.27, 0.12, 0.0)
    ctx.set_source(fg)
    ctx.arc(-22 * sc, -18 * sc, 21 * sc, 0, 2 * math.pi)
    ctx.fill()

    ctx.restore()


def composite_rgba(base: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    a = overlay_rgba[:, :, 3:4] / 255.0
    return base * (1.0 - a) + overlay_rgba[:, :, :3] * a


def load_font(filename: str, size_pt: int, axes: list[float] | None = None) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(FONTS_DIR / filename), size_pt * S)
    if axes is not None:
        font.set_variation_by_axes(axes)
    return font


def draw_leader_pin(
    ctx: cairo.Context,
    start: tuple[float, float],
    elbow: tuple[float, float],
    target: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    """Draw a crisp editorial leader pin from a callout plaque to a geographic hotspot."""
    r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
    ctx.save()
    ctx.set_source_rgba(r, g, b, 0.88)
    ctx.set_line_width(1.6 * S)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.move_to(start[0] * S, start[1] * S)
    ctx.line_to(elbow[0] * S, elbow[1] * S)
    ctx.line_to(target[0] * S, target[1] * S)
    ctx.stroke()

    # Hotspot beacon dot & outer ring
    ctx.set_source_rgba(r, g, b, 0.35)
    ctx.arc(target[0] * S, target[1] * S, 6.5 * S, 0, 2 * math.pi)
    ctx.fill()
    ctx.set_source_rgba(r, g, b, 1.0)
    ctx.arc(target[0] * S, target[1] * S, 3.0 * S, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()


def draw_editorial_callout(
    img: Image.Image,
    x: int,
    y: int,
    min_w: int,
    accent: tuple[int, int, int],
    tag: str,
    title: str,
    detail: str,
    f_tag: ImageFont.FreeTypeFont,
    f_title: ImageFont.FreeTypeFont,
    f_detail: ImageFont.FreeTypeFont,
) -> tuple[int, int, int, int]:
    """Draw a compact editorial plaque and return its logical bounding box (x, y, w, h)."""
    d_tmp = ImageDraw.Draw(img)
    pad_x = 12 * S
    pad_right = 14 * S
    pad_y = 8 * S

    max_tw = max(
        d_tmp.textbbox((0, 0), tag, font=f_tag)[2],
        d_tmp.textbbox((0, 0), title, font=f_title)[2],
        d_tmp.textbbox((0, 0), detail, font=f_detail)[2],
    )
    pw = max(min_w * S, pad_x + max_tw + pad_right)
    px, py = x * S, y * S

    h_tag = 14 * S
    h_title = 21 * S
    h_detail = 17 * S
    ph = pad_y * 2 + h_tag + 3 * S + h_title + 3 * S + h_detail

    scrim = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(scrim)
    sd.rounded_rectangle([px, py, px + pw, py + ph], radius=7 * S, fill=(5, 10, 22, 218))
    sd.rounded_rectangle([px, py, px + pw, py + ph], radius=7 * S, outline=(148, 163, 184, 48), width=S)
    sd.rounded_rectangle([px, py + 5 * S, px + 4 * S, py + ph - 5 * S], radius=2 * S, fill=(*accent, 250))
    img.alpha_composite(scrim)

    d = ImageDraw.Draw(img)
    tx = px + pad_x
    ty = py + pad_y
    d.text((tx, ty), tag, font=f_tag, fill=(*accent, 255))
    ty += h_tag + 3 * S
    d.text((tx, ty), title, font=f_title, fill=(*IVORY, 255))
    ty += h_title + 3 * S
    d.text((tx, ty), detail, font=f_detail, fill=(*SLATE_TEXT, 245))
    return x, y, pw // S, ph // S


def render_poster() -> Image.Image:
    base = render_base_globe()

    # Exact projected coordinates of the 6 world hotspots
    us_x, us_y, _ = project_lonlat(-73.0, 39.5)       # Washington / US East Coast
    eu_x, eu_y, _ = project_lonlat(4.0, 50.0)         # Western Europe / Brussels
    cn_x, cn_y, _ = project_lonlat(116.0, 39.5)       # China / Beijing
    kyiv_x, kyiv_y, _ = project_lonlat(32.5, 49.5)    # Ukraine
    cairo_x, cairo_y, _ = project_lonlat(31.2, 30.0)  # Cairo
    rt_x, rt_y, _ = project_lonlat(50.1, 26.6)        # Ras Tanura
    red_x, red_y, _ = project_lonlat(39.2, 21.5)      # Red Sea / Jeddah
    atl_x, atl_y, _ = project_lonlat(-28.0, 41.0)     # North Atlantic shipping lane
    med_x, med_y, _ = project_lonlat(14.0, 35.5)      # Central Mediterranean

    # 1. Geopolitical arcs & illustrations via pycairo
    vec_surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, SIZE, SIZE)
    ctx = cairo.Context(vec_surf)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)

    # Graticule clipped to globe
    ctx.save()
    ctx.arc(GLOBE_CX * S, GLOBE_CY * S, GLOBE_R * S, 0, 2 * math.pi)
    ctx.clip()
    ctx.set_source_rgba(0.55, 0.78, 0.98, 0.08)
    ctx.set_line_width(0.8 * S)
    for lat in range(0, 76, 15):
        pts = [project_lonlat(lon, float(lat)) for lon in range(-90, 140, 4)]
        vis = [(px * S, py * S) for px, py, pz in pts if pz > 0.02]
        if len(vis) >= 2:
            ctx.move_to(*vis[0])
            for p in vis[1:]:
                ctx.line_to(*p)
            ctx.stroke()
    for lon in range(-75, 130, 15):
        pts = [project_lonlat(float(lon), lat) for lat in range(-25, 82, 4)]
        vis = [(px * S, py * S) for px, py, pz in pts if pz > 0.02]
        if len(vis) >= 2:
            ctx.move_to(*vis[0])
            for p in vis[1:]:
                ctx.line_to(*p)
            ctx.stroke()
    ctx.restore()

    # Orbital trajectory above atmospheric limb
    draw_curved_screen_arc(ctx, (120, 232), (535, 128), (945, 246), (0.28, 0.85, 0.98, 0.46), 1.5, dashed=True)

    # Global Trade amber tariff arcs: US -> Europe -> China
    draw_great_arc(ctx, -73.0, 39.5, 4.0, 50.0, 34.0, (0.96, 0.62, 0.04, 0.78), 2.1, dashed=False)
    draw_great_arc(ctx, 4.0, 50.0, 116.0, 39.5, 46.0, (0.96, 0.62, 0.04, 0.45), 1.6, dashed=True)
    draw_tariff_chevrons(ctx, (us_x + atl_x) * 0.5, (us_y + atl_y) * 0.5 - 16, 8)
    draw_tariff_chevrons(ctx, (eu_x + cn_x) * 0.5, (eu_y + cn_y) * 0.5 - 38, 12)

    # US–NATO–Persian Gulf strategic arc
    draw_great_arc(ctx, -73.0, 39.5, 14.0, 35.5, 16.0, (0.38, 0.65, 0.98, 0.65), 1.8, dashed=True)
    draw_great_arc(ctx, 14.0, 35.5, 52.0, 29.0, 14.0, (0.38, 0.65, 0.98, 0.58), 1.8, dashed=True)

    # Red Sea maritime alert arc
    draw_great_arc(ctx, 43.3, 12.6, 39.2, 21.5, 10.0, (0.94, 0.27, 0.27, 0.75), 1.8, dashed=True)

    # 6 geographically anchored editorial illustrations
    tiangong_xy = (772.0, 198.0)
    draw_tiangong_station(ctx, tiangong_xy[0], tiangong_xy[1], -12.0)
    draw_container_ship(ctx, atl_x + 15, atl_y - 42)
    draw_ukraine_vignette(ctx, kyiv_x + 6, kyiv_y - 18)
    draw_diplomacy_carrier(ctx, atl_x + 38, med_y + 42)
    draw_cairo_viaduct(ctx, cairo_x - 8, cairo_y + 4)
    draw_ras_tanura_refinery(ctx, rt_x + 24, rt_y - 18)

    # Leader pins connecting callout plaques directly to geographic hotspots
    draw_leader_pin(ctx, (375, 182), (555, 182), (665, 194), CYAN)                   # Space -> Tiangong
    draw_leader_pin(ctx, (240, 474), (240, 444), (atl_x + 15, atl_y - 26), AMBER)     # Trade -> Atlantic lane
    draw_leader_pin(ctx, (512, 342), (542, 378), (kyiv_x + 6, kyiv_y - 2), CRIMSON)   # Ukraine -> Kyiv/Kharkiv
    draw_leader_pin(ctx, (248, 792), (248, 705), (atl_x + 38, med_y + 48), ICE_BLUE)  # Diplomacy -> Carrier/NATO arc
    draw_leader_pin(ctx, (505, 810), (558, 718), (cairo_x - 6, cairo_y + 22), ORANGE) # Cairo -> Nile/Ring Road
    draw_leader_pin(ctx, (745, 780), (735, 695), (rt_x + 10, rt_y - 8), CRIMSON)      # Middle East -> Ras Tanura

    vec_rgba = cairo_surface_to_rgba(vec_surf)
    composed = composite_rgba(base, vec_rgba)

    # Volumetric bloom around key glows
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    glow_mask = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    for gx, gy, rad, col in [
        (cairo_x - 8, cairo_y - 2, 36, (255, 130, 28)),
        (rt_x + 8, rt_y - 28, 36, (255, 118, 24)),
        (kyiv_x + 14, kyiv_y - 14, 26, (239, 68, 68)),
        (tiangong_xy[0], tiangong_xy[1], 56, (56, 189, 248)),
    ]:
        g = np.exp(-((xx - gx * S) ** 2 + (yy - gy * S) ** 2) / (2.0 * (rad * S) ** 2))
        for c in range(3):
            glow_mask[:, :, c] += g * col[c] * 0.24
    composed = np.clip(composed + glow_mask, 0.0, 255.0).astype(np.uint8)

    img = Image.fromarray(composed, mode="RGB").convert("RGBA")

    # Fonts
    f_masthead_date = load_font("Ubuntu-B.ttf", 20)
    f_masthead_title = load_font("PlayfairDisplay.ttf", 52, [850])
    f_tag = load_font("Ubuntu-B.ttf", 12)
    f_card_title = load_font("Ubuntu-B.ttf", 18)
    f_card_detail = load_font("Newsreader.ttf", 14, [520, 18])
    f_footer = load_font("Ubuntu-M.ttf", 12)

    # Top Editorial Masthead
    d = ImageDraw.Draw(img)
    d.rectangle([34 * S, 16 * S, (1024 - 34) * S, 18 * S], fill=(*AMBER, 225))
    dateline = "THURSDAY, 26 NOVEMBER 2026   •   WORLD NEWS IN ONE IMAGE"
    dbbox = d.textbbox((0, 0), dateline, font=f_masthead_date)
    dw = dbbox[2] - dbbox[0]
    d.text(((SIZE - dw) // 2, 26 * S), dateline, font=f_masthead_date, fill=(251, 191, 36, 255))

    title = "A WORLD ON THE EDGE"
    tbbox = d.textbbox((0, 0), title, font=f_masthead_title)
    tw = tbbox[2] - tbbox[0]
    d.text(((SIZE - tw) // 2 + 2, 53 * S + 2), title, font=f_masthead_title, fill=(0, 0, 0, 190))
    d.text(((SIZE - tw) // 2, 53 * S), title, font=f_masthead_title, fill=(*IVORY, 255))

    d.line([(34 * S, 120 * S), ((1024 - 34) * S, 120 * S)], fill=(148, 163, 184, 95), width=S)
    d.line([(412 * S, 120 * S), (612 * S, 120 * S)], fill=(*AMBER, 225), width=2 * S)

    # 6 Sleek Editorial Plaques anchored around the globe
    draw_editorial_callout(
        img, 32, 146, 342, CYAN,
        "SPACE  •  TIANGONG ORBIT",
        "PAKISTAN'S FIRST ASTRONAUT",
        "Muhammad Zeeshan Ali docks on Shenzhou-24",
        f_tag, f_card_title, f_card_detail,
    )

    draw_editorial_callout(
        img, 362, 268, 294, CRIMSON,
        "UKRAINE WAR",
        "WINTER GRID & CITY STRIKES",
        "100+ drones & missiles; 5+ regions in outage",
        f_tag, f_card_title, f_card_detail,
    )

    draw_editorial_callout(
        img, 28, 474, 290, AMBER,
        "GLOBAL TRADE",
        "US 14-PARTNER TARIFFS",
        "Proposed duties draw China & EU pushback",
        f_tag, f_card_title, f_card_detail,
    )

    draw_editorial_callout(
        img, 28, 792, 294, ICE_BLUE,
        "DIPLOMACY",
        "US–NATO: UKRAINE & IRAN",
        "Trump–Rutte statement; US carrier surge",
        f_tag, f_card_title, f_card_detail,
    )

    draw_editorial_callout(
        img, 356, 810, 292, ORANGE,
        "CAIRO DISASTER",
        "RING ROAD TANKER BLAZE",
        "At least 31 killed in Giza highway crash",
        f_tag, f_card_title, f_card_detail,
    )

    draw_editorial_callout(
        img, 694, 780, 298, CRIMSON,
        "MIDDLE EAST",
        "RAS TANURA & RED SEA",
        "Refinery smoke; Saudi strikes in Yemen",
        f_tag, f_card_title, f_card_detail,
    )

    # Clean dark scrim footer bar (prevents bright ocean glow behind legend text)
    footer_scrim = Image.new("RGBA", img.size, (0, 0, 0, 0))
    fd = ImageDraw.Draw(footer_scrim)
    fd.rectangle([0, 954 * S, SIZE, SIZE], fill=(4, 8, 20, 240))
    fd.line([(34 * S, 954 * S), ((1024 - 34) * S, 954 * S)], fill=(148, 163, 184, 90), width=S)
    img.alpha_composite(footer_scrim)

    d = ImageDraw.Draw(img)
    d.text((34 * S, 972 * S), "26 NOVEMBER 2026  •  SIX FRONTS ACROSS LAND, SEA & ORBIT", font=f_footer, fill=(170, 184, 204, 240))

    key_items = [
        ("ENERGY / CRISIS", AMBER),
        ("CRITICAL TENSION", CRIMSON),
        ("DIPLOMACY", ICE_BLUE),
        ("ORBIT", CYAN),
    ]
    kx = 990 * S
    for label, col in reversed(key_items):
        lb = d.textbbox((0, 0), label, font=f_footer)
        lw = lb[2] - lb[0]
        kx -= lw
        d.text((kx, 972 * S), label, font=f_footer, fill=(210, 220, 234, 245))
        kx -= 14 * S
        d.ellipse([kx, 975 * S, kx + 8 * S, 983 * S], fill=(*col, 255))
        kx -= 18 * S

    final = img.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    return final


def main() -> None:
    poster = render_poster()
    poster.save(OUT_PATH, format="PNG", optimize=True)
    size_bytes = OUT_PATH.stat().st_size
    print(f"Wrote {OUT_PATH} ({poster.size[0]}x{poster.size[1]}, {size_bytes:,} bytes)")


if __name__ == "__main__":
    main()
