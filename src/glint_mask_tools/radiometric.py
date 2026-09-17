"""MicaSense radiometric correction: raw DN → surface reflectance.

Implements the documented MicaSense conversion chain per band per capture:

  1. Subtract BlackLevel (TIFF tag).
  2. Row-gradient correction using RadiometricCalibration coefficients (a2, a3):
     p = DN - BL - a2*y - a3*y*t
  3. Vignetting correction using VignettingCenter and VignettingPolynomial:
     V(x, y) = 1 / (1 + Σ k_i · r^(i+1))
  4. Radiance conversion:
     L = V(x, y) · a1/gain · p / (t · (2^bits − 1))       [W/m²/sr/nm]
  5. DLS sun-angle correction. The XMP Irradiance value is measured on the
     DLS's own tilted plane. Correcting it to what a level, upward-facing
     sensor would read requires knowing where the sun actually is relative to
     the DLS's pointing direction (not just how much the plane is tilted) —
     see ``solar_geometry.horizontal_irradiance`` for the full model. Falls
     back to a simple ``E / (cos(pitch) · cos(roll))`` projection (which
     implicitly assumes an overhead sun) when GPS position, capture time, or
     yaw aren't available.
     Also convert from µW/cm²/nm (DLS reported units) to W/m²/nm (SI, ÷100).
  6. Reflectance:
     R = π · L / E_horizontal                              [dimensionless]

The vignetting polynomial evaluation dominates the per-pixel cost, so per-band
maps are cached the first time each (band, image size) pair is seen.

Reference: MicaSense Image Processing tutorials + imageprocessing library.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import numpy as np
import tifffile
from loguru import logger

from . import solar_geometry

if TYPE_CHECKING:
    from collections.abc import Sequence


_XMP_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "Camera": "http://pix4d.com/camera/1.0",
    "MicaSense": "http://micasense.com/MicaSense/1.0/",
}

_DEFAULT_BIT_DEPTH_MAX = 65535.0  # 2^16 − 1

# Fraction of full scale above which a pixel is treated as saturated. MicaSense
# writes 12-bit data left-shifted into 16 bits, so its real ceiling is 65520
# rather than 65535; 0.98 catches that without flagging merely-bright pixels.
SATURATION_DN_FRACTION = 0.98


@dataclass(frozen=True)
class BandRadiometry:
    """Per-band radiometric metadata needed to convert DN → reflectance."""

    band_idx: int
    black_level: float
    exposure_time_s: float
    iso: int
    radiometric_calibration: tuple[float, float, float]      # (a1, a2, a3)
    vignetting_center_px: tuple[float, float]                # (x, y)
    vignetting_polynomial: tuple[float, ...]                 # k_0 .. k_n
    irradiance_micro_W_per_cm2_per_nm: float                 # DLS raw reading
    irradiance_pitch_rad: float
    irradiance_roll_rad: float
    irradiance_yaw_rad: float | None                         # None if unavailable
    latitude_deg: float | None                               # None if GPS missing
    longitude_deg: float | None
    capture_utc: datetime | None                             # None if timestamp missing
    image_size: tuple[int, int]                              # (W, H)
    # Horizontal irradiance supplied by a flight-level model instead of this
    # capture's own DLS geometry. Set by ``irradiance.IrradianceCalibrator``
    # for captures whose sun-sensor angle makes their own reading unusable.
    horizontal_irradiance_override: float | None = None

    @property
    def gain(self) -> float:
        """Analog gain derived from ISO (ISO 100 = gain 1)."""
        return self.iso / 100.0

    @property
    def irradiance_W_per_m2_per_nm(self) -> float:
        """DLS reading in SI units (µW/cm²/nm ÷ 100 = W/m²/nm)."""
        return self.irradiance_micro_W_per_cm2_per_nm / 100.0

    @property
    def irradiance_horizontal_W_per_m2_per_nm(self) -> float:
        """DLS reading corrected to what a level, upward-facing sensor would read.

        A value supplied by a flight-level model (see
        ``irradiance.IrradianceCalibrator``) wins over anything derivable from
        this capture alone, because the DLS's own reading is unusable once the
        sun passes behind its dome.

        Otherwise uses the full sun-angle model (solar position from GPS+time,
        and the 3D angle between the sun and the DLS's actual pointing
        direction) when yaw, GPS position, and capture time are all available.
        Falls back to a simple overhead-sun tilt projection otherwise —
        adequate for small tilts, but increasingly wrong as tilt grows if the
        sun isn't actually near zenith.
        """
        if self.horizontal_irradiance_override is not None:
            return self.horizontal_irradiance_override

        if (
            self.irradiance_yaw_rad is not None
            and self.latitude_deg is not None
            and self.longitude_deg is not None
            and self.capture_utc is not None
        ):
            sun = solar_geometry.solar_position(self.latitude_deg, self.longitude_deg, self.capture_utc)
            return solar_geometry.horizontal_irradiance(
                self.irradiance_W_per_m2_per_nm,
                sun,
                self.irradiance_yaw_rad,
                self.irradiance_pitch_rad,
                self.irradiance_roll_rad,
            )

        logger.warning(
            f"Band {self.band_idx}: missing yaw/GPS/timestamp for proper DLS sun-angle "
            "correction; falling back to overhead-sun tilt projection."
        )
        tilt_cos = np.cos(self.irradiance_pitch_rad) * np.cos(self.irradiance_roll_rad)
        tilt_cos = max(tilt_cos, 0.05)  # avoid divide-by-near-zero for extreme tilts
        return self.irradiance_W_per_m2_per_nm / tilt_cos


class _VignetteMapCache:
    """Cache per-band vignetting maps to avoid recomputing them each capture."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, tuple[float, ...], tuple[float, float], tuple[int, int]], np.ndarray] = {}

    def get(self, meta: BandRadiometry) -> np.ndarray:
        key = (
            meta.band_idx,
            tuple(meta.vignetting_polynomial),
            meta.vignetting_center_px,
            meta.image_size,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        v_map = _compute_vignetting_map(meta)
        self._cache[key] = v_map
        return v_map


_vignette_cache = _VignetteMapCache()


def _compute_vignetting_map(meta: BandRadiometry) -> np.ndarray:
    """Return V(x, y) = 1 / (1 + Σ k_i · r^(i+1))."""
    w, h = meta.image_size
    cx, cy = meta.vignetting_center_px
    x = np.arange(w, dtype=np.float32) - cx
    y = np.arange(h, dtype=np.float32)[:, None] - cy
    r = np.hypot(x, y)
    poly = np.ones_like(r, dtype=np.float32)
    for i, k in enumerate(meta.vignetting_polynomial):
        poly = poly + np.float32(k) * (r ** (i + 1))
    # Guard against divide-by-zero on degenerate coefficients.
    poly = np.where(np.abs(poly) < 1e-6, np.float32(1.0), poly)
    return (1.0 / poly).astype(np.float32)


def dn_to_reflectance(dn: np.ndarray, meta: BandRadiometry) -> np.ndarray:
    """Convert a raw MicaSense DN image to surface reflectance.

    Parameters
    ----------
    dn
        Raw pixel values as float (H, W). Values are on the sensor's DN scale.
    meta
        Radiometric metadata for this band + capture.

    Returns
    -------
    Reflectance array (H, W), roughly [0, 1] for diffuse surfaces and possibly
    >1 for specular / saturating features (glint, wave crests).

    """
    a1, a2, a3 = meta.radiometric_calibration
    t = meta.exposure_time_s
    g = meta.gain

    # 1 + 2. Black level + row gradient correction.
    h, w = dn.shape
    y = np.arange(h, dtype=np.float64)[:, None]
    p = dn.astype(np.float64) - meta.black_level - a2 * y - a3 * y * t
    # Radiance is non-negative; clip pre-conversion noise.
    p = np.maximum(p, 0.0)

    # 3. Vignetting correction (cached per band).
    v_map = _vignette_cache.get(meta)

    # 4. Radiance in W/m²/sr/nm.
    l = v_map * (a1 / g) * p / (t * _DEFAULT_BIT_DEPTH_MAX)

    # 5+6. Reflectance = π · L / E_horizontal.
    e_horizontal = meta.irradiance_horizontal_W_per_m2_per_nm
    if e_horizontal <= 0:
        # Malformed DLS reading; return the raw radiance normalized by max
        # so downstream at least gets a value in [0, ~1].
        logger.warning(
            f"Band {meta.band_idx} has non-positive DLS irradiance "
            f"({meta.irradiance_micro_W_per_cm2_per_nm:.3f} µW/cm²/nm); "
            "skipping reflectance conversion, returning normalized radiance."
        )
        max_l = float(np.max(l)) or 1.0
        return (l / max_l).astype(np.float32)
    return (np.pi * l / e_horizontal).astype(np.float32)


def saturation_reflectance_ceiling(meta: BandRadiometry) -> float:
    """Highest reflectance this capture can express: what a saturated pixel converts to.

    Radiance is ``DN / (exposure * gain)``, so a clipped pixel's *converted*
    value depends entirely on the exposure the camera happened to choose. A long
    auto-exposure therefore pins the top of the reflectance scale low — on real
    flights low enough that a threshold sits above it and can never trigger, no
    matter how bright the glint actually was. Comparing a threshold against this
    ceiling says whether the threshold is reachable at all for a capture.

    Computed at the vignetting centre (V = 1); off-centre pixels have V > 1 and
    so a slightly higher ceiling, making this the conservative estimate.
    """
    e_horizontal = meta.irradiance_horizontal_W_per_m2_per_nm
    if e_horizontal <= 0:
        return float("inf")
    a1, _, _ = meta.radiometric_calibration
    saturated_dn = SATURATION_DN_FRACTION * _DEFAULT_BIT_DEPTH_MAX
    radiance = (a1 / meta.gain) * (saturated_dn - meta.black_level) / (meta.exposure_time_s * _DEFAULT_BIT_DEPTH_MAX)
    return float(np.pi * radiance / e_horizontal)


# ---------------------------------------------------------------------------
# XMP + TIFF metadata parsing
# ---------------------------------------------------------------------------


def _find_first(root: ET.Element, path: str) -> ET.Element | None:
    """Search every rdf:Description block for the first matching element.

    MicaSense splits XMP fields across two namespaces (Camera / MicaSense), each
    in its own rdf:Description. Callers pass the fully-qualified tag path
    (e.g. "MicaSense:RadiometricCalibration") and get back whichever block
    contains it, without needing to know which namespace holds each field.
    """
    return root.find(f".//{path}", _XMP_NS)


def _text_of(root: ET.Element, path: str) -> str | None:
    el = _find_first(root, path)
    return el.text.strip() if el is not None and el.text is not None else None


def _seq_floats(root: ET.Element, path: str) -> list[float] | None:
    el = _find_first(root, path)
    if el is None:
        return None
    seq = el.find("rdf:Seq", _XMP_NS)
    if seq is None:
        return None
    return [float(li.text) for li in seq.findall("rdf:li", _XMP_NS) if li.text is not None]


def _rational_to_float(value: object) -> float | None:
    """Convert a tifffile EXIF rational value to a float.

    Handles a single (num, den) pair, a flat tuple of N (num, den) pairs
    (as tifffile returns for GPS DMS triplets), or an already-resolved number.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, tuple):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):  # noqa: PLR2004
            num, den = value
            return float(num) / float(den) if den else None
    return None


def _gps_dms_to_decimal(value: object) -> float | None:
    """Convert a tifffile GPS DMS tuple (6 raw ints: d_num,d_den,m_num,m_den,s_num,s_den) to decimal degrees."""
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 6:  # noqa: PLR2004
        d = value[0] / value[1] if value[1] else 0.0
        m = value[2] / value[3] if value[3] else 0.0
        s = value[4] / value[5] if value[5] else 0.0
        return d + m / 60.0 + s / 3600.0
    if isinstance(value, tuple) and len(value) == 3:  # noqa: PLR2004
        # Already-resolved (deg, min, sec) floats.
        d, m, s = value
        return float(d) + float(m) / 60.0 + float(s) / 3600.0
    return None


def _parse_capture_utc(exif: dict) -> datetime | None:
    """Parse EXIF DateTimeOriginal + SubSecTime as a UTC datetime.

    MicaSense records DateTimeOriginal directly in UTC (confirmed against the
    reference imageprocessing library's own ``utc_time()``, which localizes
    the raw string to UTC without any timezone conversion) — so no timezone
    adjustment is applied here.
    """
    raw = exif.get("DateTimeOriginal")
    if not raw:
        return None
    try:
        dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    subsec_raw = exif.get("SubsecTime") or exif.get("SubSecTime")
    if subsec_raw:
        try:
            frac = float(f"0.{subsec_raw}")
            dt = dt.replace(microsecond=int(round(frac * 1e6)))
        except ValueError:
            pass
    return dt.replace(tzinfo=timezone.utc)


def parse_micasense_radiometry(path: str | Path, band_idx: int) -> BandRadiometry | None:
    """Parse per-band radiometric metadata from a MicaSense TIFF.

    Returns None if any required field is missing (non-MicaSense or old firmware).
    GPS position, capture timestamp, and yaw are parsed on a best-effort basis —
    their absence doesn't fail the whole capture, it just falls back to the
    simpler overhead-sun tilt projection for the DLS correction (see
    ``BandRadiometry.irradiance_horizontal_W_per_m2_per_nm``).
    """
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        h, w = page.shape[:2]

        # BlackLevel comes from a standard TIFF tag, not XMP.
        bl_tag = page.tags.get("BlackLevel")
        if bl_tag is None:
            return None
        black_level = float(bl_tag.value[0] if hasattr(bl_tag.value, "__getitem__") else bl_tag.value)

        # EXIF (ExposureTime + ISO + capture time).
        exif_tag = page.tags.get("ExifTag")
        if exif_tag is None:
            return None
        exif = exif_tag.value
        et = exif.get("ExposureTime")
        if et is None:
            return None
        exposure_time_s = et[0] / et[1] if isinstance(et, tuple) else float(et)
        iso = int(exif.get("ISOSpeed") or exif.get("PhotographicSensitivity") or 100)
        capture_utc = _parse_capture_utc(exif)

        # GPS (lat/lon) — best effort.
        latitude_deg = longitude_deg = None
        gps_tag = page.tags.get("GPSTag")
        if gps_tag is not None:
            gps = gps_tag.value
            lat = _gps_dms_to_decimal(gps.get("GPSLatitude"))
            lon = _gps_dms_to_decimal(gps.get("GPSLongitude"))
            if lat is not None and gps.get("GPSLatitudeRef") == "S":
                lat = -lat
            if lon is not None and gps.get("GPSLongitudeRef") == "W":
                lon = -lon
            latitude_deg, longitude_deg = lat, lon

        # XMP (radiometric coefficients, vignetting, DLS irradiance + pose).
        xmp = page.tags.get("XMP")
        if xmp is None:
            return None
        xmp_str = xmp.value.decode("utf-8", errors="ignore") if isinstance(xmp.value, bytes) else str(xmp.value)

    try:
        root = ET.fromstring(xmp_str)
    except ET.ParseError:
        return None

    # MicaSense splits XMP fields between Camera (Pix4D) and MicaSense namespaces.
    # RadiometricCalibration is in the MicaSense block; VignettingCenter,
    # VignettingPolynomial, and Irradiance* are in the Camera block. Search
    # across every rdf:Description so we don't care which is which.
    rc = _seq_floats(root, "MicaSense:RadiometricCalibration")
    if rc is None or len(rc) < 3:  # noqa: PLR2004
        return None
    vc = _seq_floats(root, "Camera:VignettingCenter")
    if vc is None or len(vc) < 2:  # noqa: PLR2004
        return None
    vp = _seq_floats(root, "Camera:VignettingPolynomial")
    if vp is None:
        return None
    irr_text = _text_of(root, "Camera:Irradiance")
    if irr_text is None:
        return None
    pitch_text = _text_of(root, "Camera:IrradiancePitch") or "0"
    roll_text = _text_of(root, "Camera:IrradianceRoll") or "0"
    yaw_text = _text_of(root, "Camera:IrradianceYaw")
    yaw_rad = float(np.deg2rad(float(yaw_text))) if yaw_text is not None else None

    return BandRadiometry(
        band_idx=band_idx,
        black_level=black_level,
        exposure_time_s=exposure_time_s,
        iso=iso,
        radiometric_calibration=(rc[0], rc[1], rc[2]),
        vignetting_center_px=(vc[0], vc[1]),
        vignetting_polynomial=tuple(vp),
        irradiance_micro_W_per_cm2_per_nm=float(irr_text),
        irradiance_pitch_rad=float(np.deg2rad(float(pitch_text))),
        irradiance_roll_rad=float(np.deg2rad(float(roll_text))),
        irradiance_yaw_rad=yaw_rad,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        capture_utc=capture_utc,
        image_size=(w, h),
    )


def parse_capture_radiometry(paths: Sequence[str | Path]) -> list[BandRadiometry] | None:
    """Parse radiometric metadata for every band of a MicaSense capture.

    Returns None if any band fails to parse (treat the whole capture as unusable
    for reflectance and fall back to bit-depth normalization).
    """
    metadata: list[BandRadiometry] = []
    for i, p in enumerate(paths):
        m = parse_micasense_radiometry(p, i)
        if m is None:
            return None
        metadata.append(m)
    return metadata


def apply_reflectance_conversion(
    img: np.ndarray,
    metadata: Sequence[BandRadiometry],
) -> np.ndarray:
    """Convert a multi-band DN image to reflectance using per-band metadata.

    Parameters
    ----------
    img
        Raw DN image stacked as (H, W, C).
    metadata
        Per-band ``BandRadiometry`` in the same order as the image's C axis.

    Returns
    -------
    Reflectance array (H, W, C).

    """
    if img.shape[2] != len(metadata):
        msg = f"img has {img.shape[2]} bands but metadata has {len(metadata)}"
        raise ValueError(msg)
    out = np.empty_like(img, dtype=np.float32)
    for i, m in enumerate(metadata):
        out[:, :, i] = dn_to_reflectance(img[:, :, i], m)
    return out
