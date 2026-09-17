"""Flight-level stabilization of DLS downwelling irradiance.

Why this exists: the DLS is bolted to the airframe, so its dome tilts with the
aircraft. Converting its reading to the horizontal-plane irradiance that
reflectance needs requires projecting the direct solar beam, and that
projection is only conditioned while the sun is reasonably close to the dome's
axis (see ``solar_geometry.MAX_RELIABLE_SUN_SENSOR_ANGLE_DEG``). On a survey
flown late in the day, the reciprocal legs of a lawnmower pattern put the sun
behind the dome on every other leg — the sun-sensor angle crosses 90°, the
direct beam is unmeasurable, and the per-capture estimate becomes meaningless.
Because reflectance is ``pi * L / E``, an E that is 10x too small turns every
pixel into apparent glint, and an E that is 10x too large turns glint
invisible. Whole legs of a flight come out fully masked or not masked at all.

The fix leans on physics the DLS can't ruin: under stable illumination the
*horizontal* downwelling irradiance is a smooth function of time (it tracks
solar elevation), no matter how the aircraft is oriented. So:

  1. Sweep a sample of captures across the flight and compute each one's
     sun-sensor angle.
  2. Keep only the captures whose geometry is well-conditioned, and build a
     per-band irradiance-versus-time series from those (median-filtered, so a
     single bad reading can't move it).
  3. Captures with unusable geometry get their irradiance interpolated from
     that series instead of from their own DLS reading.

Captures whose own geometry is fine keep their own measured value, so this is
a no-op on flights that never tilt away from the sun, and the interpolated and
measured values agree by construction where the two regimes meet.

If a flight has too few well-conditioned captures to build a series (e.g. the
DLS pointed away from the sun for its entire duration), calibration is skipped
and each capture falls back to the clamped per-capture model.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Callable

import numpy as np
from loguru import logger

from . import solar_geometry
from .radiometric import saturation_reflectance_ceiling

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .radiometric import BandRadiometry

# Captures sampled during the calibration sweep. The irradiance series is
# smooth in time, so a few hundred samples pin it down regardless of how many
# thousand captures the flight holds; each sample costs one metadata parse.
DEFAULT_MAX_CALIBRATION_CAPTURES = 250

# Width of the running median applied to the reliable samples. Wide enough to
# reject a lone bad reading, narrow enough to follow real illumination drift
# (thin cloud, solar elevation change) over a flight.
_MEDIAN_FILTER_WINDOW = 5

# Below this many well-conditioned captures there isn't enough signal to
# interpolate from, and calibration is skipped entirely.
_MIN_RELIABLE_CAPTURES = 3


def sun_sensor_angle_deg(meta: BandRadiometry) -> float | None:
    """Return the angle between the sun and the DLS dome axis, in degrees.

    Returns None when the capture lacks the yaw, GPS position, or timestamp
    needed to place the sun — those captures can't be assessed, so callers
    treat them as unreliable.
    """
    if (
        meta.irradiance_yaw_rad is None
        or meta.latitude_deg is None
        or meta.longitude_deg is None
        or meta.capture_utc is None
    ):
        return None
    sun = solar_geometry.solar_position(meta.latitude_deg, meta.longitude_deg, meta.capture_utc)
    angle = solar_geometry.sun_sensor_angle(
        sun,
        meta.irradiance_yaw_rad,
        meta.irradiance_pitch_rad,
        meta.irradiance_roll_rad,
    )
    return math.degrees(angle)


def geometry_is_reliable(meta: BandRadiometry, max_angle_deg: float) -> bool:
    """Whether this capture's DLS geometry supports a trustworthy irradiance estimate."""
    angle = sun_sensor_angle_deg(meta)
    return angle is not None and angle <= max_angle_deg


class IrradianceCalibrator:
    """Builds a per-band irradiance-versus-time model from well-conditioned captures.

    Usage mirrors ``BandAligner``: call ``calibrate`` once with the flight's
    capture paths, then ``apply`` per capture to get metadata with a
    trustworthy horizontal irradiance attached.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_sun_sensor_angle_deg: float = solar_geometry.MAX_RELIABLE_SUN_SENSOR_ANGLE_DEG,
        max_calibration_captures: int = DEFAULT_MAX_CALIBRATION_CAPTURES,
    ) -> None:
        """Create a new IrradianceCalibrator.

        Parameters
        ----------
        enabled
            When False, ``calibrate`` and ``apply`` are no-ops and every
            capture keeps its own per-capture irradiance estimate.
        max_sun_sensor_angle_deg
            Captures whose sun-sensor angle exceeds this are treated as
            unusable and have their irradiance interpolated instead.
        max_calibration_captures
            Upper bound on captures sampled during the calibration sweep.

        """
        self.enabled = enabled
        self.max_sun_sensor_angle_deg = max_sun_sensor_angle_deg
        self.max_calibration_captures = max_calibration_captures
        self._times: np.ndarray | None = None
        self._irradiance: np.ndarray | None = None  # (n_samples, n_bands)
        self._ceilings: np.ndarray | None = None  # (n_samples, n_bands)
        self._attempted = False

    @property
    def is_calibrated(self) -> bool:
        """Whether a usable irradiance series was built."""
        return self._times is not None

    @property
    def calibration_attempted(self) -> bool:
        """Whether a calibration sweep has already run (successful or not)."""
        return self._attempted

    def calibrate(
        self,
        capture_paths: Iterable[list[str] | str],
        read_metadata_fn: Callable[[list[str] | str], Sequence[BandRadiometry] | None],
    ) -> None:
        """Build the irradiance-versus-time series for this flight.

        Parameters
        ----------
        capture_paths
            Every capture in the job, as produced by ``ImageLoader.paths``.
        read_metadata_fn
            Reads per-band radiometry for one capture, returning None for
            imagery without MicaSense radiometric metadata (in which case
            calibration is silently skipped — there is no irradiance to model).

        """
        if not self.enabled or self._attempted:
            return
        self._attempted = True

        paths = list(capture_paths)
        if not paths:
            return

        times, samples, ceilings, n_parsed = self._sweep(paths, read_metadata_fn)
        n_reliable = len(times)

        if ceilings:
            self._ceilings = np.asarray(ceilings)

        if n_parsed == 0:
            # Not MicaSense imagery, or no radiometric metadata: nothing to do.
            return

        if n_reliable < _MIN_RELIABLE_CAPTURES:
            logger.warning(
                f"DLS irradiance calibration skipped: only {n_reliable} of {n_parsed} sampled captures had "
                f"a sun-sensor angle within {self.max_sun_sensor_angle_deg:.0f}deg. Every capture will fall back "
                "to the clamped per-capture estimate, which may under- or over-mask when the aircraft tilts "
                "away from the sun."
            )
            return

        order = np.argsort(np.asarray(times))
        self._times = np.asarray(times)[order]
        self._irradiance = _running_median(np.asarray(samples)[order], _MEDIAN_FILTER_WINDOW)

        logger.info(
            f"DLS irradiance calibrated from {n_reliable}/{n_parsed} sampled captures with usable sun-sensor "
            f"geometry (<={self.max_sun_sensor_angle_deg:.0f}deg). Captures outside that will have their "
            "irradiance interpolated from this series."
        )

    def _sweep(
        self,
        paths: list[list[str] | str],
        read_metadata_fn: Callable[[list[str] | str], Sequence[BandRadiometry] | None],
    ) -> tuple[list[float], list[list[float]], list[list[float]], int]:
        """Sample captures across the flight, returning (times, irradiance, ceilings, n_parsed).

        Times and irradiance cover only the captures with usable geometry;
        ceilings cover every capture that parsed, since threshold headroom is
        worth reporting regardless of where the sun was.
        """
        times: list[float] = []
        samples: list[list[float]] = []
        ceilings: list[list[float]] = []
        n_parsed = 0
        n_unreadable = 0

        for path in _evenly_spaced(paths, self.max_calibration_captures):
            try:
                metadata = read_metadata_fn(path)
            except Exception as exc:
                # A single corrupt band file shouldn't take down the whole job
                # before any masking happens — drop the capture from the sweep
                # and let the masking pass report it per-capture as usual.
                n_unreadable += 1
                logger.warning(f"Skipping {path} during irradiance calibration: {exc}")
                continue
            if not metadata:
                continue
            n_parsed += 1
            ceilings.append([saturation_reflectance_ceiling(m) for m in metadata])
            capture_utc = metadata[0].capture_utc
            if capture_utc is None:
                continue
            if not all(geometry_is_reliable(m, self.max_sun_sensor_angle_deg) for m in metadata):
                continue
            times.append(capture_utc.timestamp())
            samples.append([m.irradiance_horizontal_W_per_m2_per_nm for m in metadata])

        if n_unreadable:
            logger.warning(f"{n_unreadable} sampled capture(s) could not be read during irradiance calibration.")

        return times, samples, ceilings, n_parsed

    def warn_on_unreachable_thresholds(self, thresholds: Sequence[float]) -> None:
        """Log a warning for any band whose threshold the data can't reach.

        A clipped pixel converts to ``pi * L_sat / E``, and ``L_sat`` is divided
        by exposure time, so a long auto-exposure pushes a capture's maximum
        expressible reflectance down. When a threshold sits above that ceiling
        no pixel can trigger it however bright the scene was, and the capture
        yields an empty mask with nothing in the output to explain why.
        """
        if self._ceilings is None or not len(thresholds):
            return
        n_samples, n_bands = self._ceilings.shape
        if n_bands != len(thresholds):
            return

        unreachable = self._ceilings < np.asarray(thresholds, dtype=float)
        any_band = unreachable.any(axis=1)
        if not any_band.any():
            return

        share = 100.0 * any_band.mean()
        worst = [
            f"band {b} (threshold {thresholds[b]:g}, reachable on "
            f"{100.0 * (~unreachable[:, b]).mean():.0f}% of captures)"
            for b in range(n_bands)
            if unreachable[:, b].any()
        ]
        logger.warning(
            f"Threshold unreachable on {share:.0f}% of {n_samples} sampled captures: the sensor saturates "
            f"below the requested value, so those captures can only be masked via clipped pixels. "
            f"Affected: {'; '.join(worst)}. This is an exposure limit, not a glint measurement — lower the "
            "threshold, or fix the camera's exposure so glint stays inside the sensor's range."
        )

    def apply(self, metadata: Sequence[BandRadiometry]) -> list[BandRadiometry]:
        """Return this capture's metadata with a trustworthy irradiance attached.

        Bands whose own geometry is well-conditioned are returned untouched.
        The rest get an override interpolated from the calibrated series.
        """
        if not self.enabled or not self.is_calibrated or not metadata:
            return list(metadata)

        capture_utc = metadata[0].capture_utc
        if capture_utc is None:
            return list(metadata)
        t = capture_utc.timestamp()

        out: list[BandRadiometry] = []
        for band_idx, m in enumerate(metadata):
            if geometry_is_reliable(m, self.max_sun_sensor_angle_deg):
                out.append(m)
                continue
            # np.interp holds the endpoint value outside the calibrated range,
            # which is what we want for captures before/after the sampled span.
            interpolated = float(np.interp(t, self._times, self._irradiance[:, band_idx]))
            out.append(dataclasses.replace(m, horizontal_irradiance_override=interpolated))
        return out


def _evenly_spaced(items: list, limit: int) -> list:
    """Return at most ``limit`` items spread evenly across the list."""
    if limit <= 0 or len(items) <= limit:
        return items
    idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in dict.fromkeys(idx.tolist())]


def _running_median(values: np.ndarray, window: int) -> np.ndarray:
    """Median-filter each column of (n_samples, n_bands), clamping at the edges."""
    n = values.shape[0]
    half = window // 2
    out = np.empty_like(values)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.median(values[lo:hi], axis=0)
    return out
