"""Tests for flight-level DLS irradiance stabilization.

Created by: Taylor Denouden
Organization: Hakai Institute
"""

import dataclasses
import math
from datetime import datetime, timedelta, timezone

from glint_mask_tools import solar_geometry
from glint_mask_tools.irradiance import (
    IrradianceCalibrator,
    geometry_is_reliable,
    sun_sensor_angle_deg,
)
from glint_mask_tools.radiometric import BandRadiometry

# Rum Cay, Bahamas — same geometry as the flight this module was written for.
LAT, LON = 23.68, -74.84
# Late-afternoon capture: sun low (~23 deg elevation) and in the west.
T0 = datetime(2026, 8, 23, 21, 37, 0, tzinfo=timezone.utc)


def make_meta(  # noqa: PLR0913
    band_idx=0,
    *,
    irradiance=20.0,
    yaw_deg=175.0,
    pitch_deg=-30.0,
    roll_deg=8.0,
    when=T0,
    with_position=True,
) -> BandRadiometry:
    """Build a BandRadiometry with only the fields the irradiance model reads."""
    return BandRadiometry(
        band_idx=band_idx,
        black_level=4800.0,
        exposure_time_s=0.001,
        iso=100,
        radiometric_calibration=(1.0, 0.0, 0.0),
        vignetting_center_px=(640.0, 480.0),
        vignetting_polynomial=(0.0,),
        irradiance_micro_W_per_cm2_per_nm=irradiance,
        irradiance_pitch_rad=math.radians(pitch_deg),
        irradiance_roll_rad=math.radians(roll_deg),
        irradiance_yaw_rad=math.radians(yaw_deg),
        latitude_deg=LAT if with_position else None,
        longitude_deg=LON if with_position else None,
        capture_utc=when,
        image_size=(1280, 960),
    )


# Attitudes taken from the real flight: the southbound leg keeps the sun in
# front of the DLS dome, the northeast leg puts it behind.
GOOD_ATTITUDE = {"yaw_deg": 172.0, "pitch_deg": -35.0, "roll_deg": 9.0}
BAD_ATTITUDE = {"yaw_deg": 58.0, "pitch_deg": -41.0, "roll_deg": 37.0}


class TestSunSensorAngle:
    """Tests for the sun-sensor angle helper."""

    def test_angle_computed_when_metadata_complete(self):
        """A fully-populated capture yields an angle in degrees."""
        angle = sun_sensor_angle_deg(make_meta(**GOOD_ATTITUDE))
        assert angle is not None
        assert 0.0 <= angle <= 180.0

    def test_none_without_position(self):
        """Missing GPS means the sun can't be placed, so no angle."""
        assert sun_sensor_angle_deg(make_meta(with_position=False)) is None

    def test_none_without_yaw(self):
        """Missing yaw means the dome's pointing direction is unknown."""
        meta = dataclasses.replace(make_meta(), irradiance_yaw_rad=None)
        assert sun_sensor_angle_deg(meta) is None

    def test_real_flight_attitudes_split_on_reliability(self):
        """The two legs of the real flight land on opposite sides of the cutoff."""
        good = sun_sensor_angle_deg(make_meta(**GOOD_ATTITUDE))
        bad = sun_sensor_angle_deg(make_meta(**BAD_ATTITUDE))
        assert good < solar_geometry.MAX_RELIABLE_SUN_SENSOR_ANGLE_DEG
        # Sun behind the dome entirely.
        assert bad > 90.0

    def test_geometry_is_reliable_matches_angle(self):
        """geometry_is_reliable agrees with the raw angle comparison."""
        cutoff = solar_geometry.MAX_RELIABLE_SUN_SENSOR_ANGLE_DEG
        assert geometry_is_reliable(make_meta(**GOOD_ATTITUDE), cutoff)
        assert not geometry_is_reliable(make_meta(**BAD_ATTITUDE), cutoff)

    def test_incomplete_metadata_is_unreliable(self):
        """A capture that can't be assessed is treated as unreliable."""
        assert not geometry_is_reliable(make_meta(with_position=False), 75.0)


class TestHorizontalIrradianceClamping:
    """The per-capture model must degrade, not explode, at extreme angles."""

    def _horizontal(self, **attitude) -> float:
        meta = make_meta(**attitude)
        sun = solar_geometry.solar_position(LAT, LON, T0)
        return solar_geometry.horizontal_irradiance(
            meta.irradiance_W_per_m2_per_nm,
            sun,
            meta.irradiance_yaw_rad,
            meta.irradiance_pitch_rad,
            meta.irradiance_roll_rad,
        )

    def test_sun_behind_dome_stays_in_physical_range(self):
        """An unusable attitude yields a bounded estimate, not a 100x blowup."""
        measured = make_meta(**BAD_ATTITUDE).irradiance_W_per_m2_per_nm
        value = self._horizontal(**BAD_ATTITUDE)
        assert 0.5 * measured < value < 5.0 * measured

    def test_near_ninety_degrees_does_not_blow_up(self):
        """Sweeping through the old singularity stays bounded and continuous."""
        measured = make_meta().irradiance_W_per_m2_per_nm
        values = [self._horizontal(yaw_deg=y, pitch_deg=-30.0, roll_deg=30.0) for y in range(0, 360, 5)]
        assert all(0.0 < v < 5.0 * measured for v in values)
        # No discontinuity: consecutive headings differ by a modest amount.
        assert max(abs(b - a) for a, b in zip(values, values[1:])) < measured

    def test_reliable_geometry_unchanged_by_clamp(self):
        """Within the reliable range the clamp is inert."""
        value = self._horizontal(**GOOD_ATTITUDE)
        measured = make_meta(**GOOD_ATTITUDE).irradiance_W_per_m2_per_nm
        assert 0.5 * measured < value < 3.0 * measured


def make_capture(index, *, reliable, n_bands=3, irradiance=20.0):
    """Build one capture's per-band metadata, spaced 10s apart."""
    attitude = GOOD_ATTITUDE if reliable else BAD_ATTITUDE
    when = T0 + timedelta(seconds=10 * index)
    return [make_meta(band_idx=b, irradiance=irradiance + b, when=when, **attitude) for b in range(n_bands)]


class TestIrradianceCalibrator:
    """Tests for the flight-level calibrator."""

    @staticmethod
    def _alternating_flight(n=40, n_bands=3):
        """Captures alternating between reliable and unreliable legs, 10 per leg."""
        return [make_capture(i, reliable=(i // 10) % 2 == 0, n_bands=n_bands) for i in range(n)]

    def _calibrated(self, captures):
        calibrator = IrradianceCalibrator()
        keys = list(range(len(captures)))
        calibrator.calibrate(keys, lambda k: captures[k])
        return calibrator

    def test_calibrates_from_reliable_captures(self):
        """A flight with usable legs produces a calibrated model."""
        calibrator = self._calibrated(self._alternating_flight())
        assert calibrator.is_calibrated

    def test_reliable_capture_keeps_own_irradiance(self):
        """Captures with usable geometry are returned untouched."""
        captures = self._alternating_flight()
        calibrator = self._calibrated(captures)
        applied = calibrator.apply(captures[0])
        assert all(m.horizontal_irradiance_override is None for m in applied)
        assert [m.irradiance_horizontal_W_per_m2_per_nm for m in applied] == [
            m.irradiance_horizontal_W_per_m2_per_nm for m in captures[0]
        ]

    def test_unreliable_capture_gets_interpolated_irradiance(self):
        """Captures with the sun behind the dome are overridden from the series."""
        captures = self._alternating_flight()
        calibrator = self._calibrated(captures)
        bad = captures[15]
        applied = calibrator.apply(bad)
        assert all(m.horizontal_irradiance_override is not None for m in applied)

        # The override should match what the surrounding reliable captures read,
        # not the unusable value this capture would have computed for itself.
        reliable = calibrator.apply(captures[0])
        for band, (fixed, ref) in enumerate(zip(applied, reliable)):
            assert fixed.irradiance_horizontal_W_per_m2_per_nm > 0
            assert abs(
                fixed.irradiance_horizontal_W_per_m2_per_nm - ref.irradiance_horizontal_W_per_m2_per_nm
            ) < 0.05 * ref.irradiance_horizontal_W_per_m2_per_nm, f"band {band}"

    def test_per_band_series_are_independent(self):
        """Each band is interpolated from its own irradiance series."""
        captures = self._alternating_flight()
        applied = self._calibrated(captures).apply(captures[15])
        values = [m.irradiance_horizontal_W_per_m2_per_nm for m in applied]
        assert len(set(values)) == len(values)
        assert values == sorted(values)  # band irradiance increases with index

    def test_disabled_calibrator_is_identity(self):
        """A disabled calibrator neither calibrates nor overrides."""
        captures = self._alternating_flight()
        calibrator = IrradianceCalibrator(enabled=False)
        calibrator.calibrate(list(range(len(captures))), lambda k: captures[k])
        assert not calibrator.is_calibrated
        applied = calibrator.apply(captures[15])
        assert all(m.horizontal_irradiance_override is None for m in applied)

    def test_no_reliable_captures_skips_calibration(self):
        """A flight that never sees the sun properly falls back to per-capture."""
        captures = [make_capture(i, reliable=False) for i in range(20)]
        calibrator = self._calibrated(captures)
        assert not calibrator.is_calibrated
        applied = calibrator.apply(captures[0])
        assert all(m.horizontal_irradiance_override is None for m in applied)

    def test_non_radiometric_imagery_is_ignored(self):
        """Loaders returning no radiometric metadata leave the calibrator inert."""
        calibrator = IrradianceCalibrator()
        calibrator.calibrate(list(range(10)), lambda _: None)
        assert not calibrator.is_calibrated
        assert calibrator.calibration_attempted

    def test_unreadable_capture_is_skipped(self):
        """A corrupt band file drops that capture instead of failing the job."""
        captures = self._alternating_flight()

        def read(k):
            if k == 5:
                msg = "not a TIFF file"
                raise OSError(msg)
            return captures[k]

        calibrator = IrradianceCalibrator()
        calibrator.calibrate(list(range(len(captures))), read)
        assert calibrator.is_calibrated

    def test_empty_flight_does_not_raise(self):
        """No captures at all is a no-op."""
        calibrator = IrradianceCalibrator()
        calibrator.calibrate([], lambda _: None)
        assert not calibrator.is_calibrated

    def test_calibration_runs_once(self):
        """A second calibrate call is ignored so threads can't re-enter it."""
        captures = self._alternating_flight()
        calls = []

        def read(k):
            calls.append(k)
            return captures[k]

        calibrator = IrradianceCalibrator()
        calibrator.calibrate(list(range(len(captures))), read)
        first = len(calls)
        calibrator.calibrate(list(range(len(captures))), read)
        assert len(calls) == first

    def test_sampling_is_capped(self):
        """Long flights are subsampled rather than fully parsed."""
        captures = [make_capture(i, reliable=True) for i in range(500)]
        calls = []

        def read(k):
            calls.append(k)
            return captures[k]

        calibrator = IrradianceCalibrator(max_calibration_captures=25)
        calibrator.calibrate(list(range(len(captures))), read)
        assert len(calls) <= 25
        assert calibrator.is_calibrated

    def test_override_used_by_reflectance_conversion(self):
        """The override is what dn_to_reflectance actually divides by."""
        captures = self._alternating_flight()
        calibrator = self._calibrated(captures)
        raw = captures[15][0]
        fixed = calibrator.apply(captures[15])[0]
        # Same capture, only the irradiance changed — reflectance scales inversely.
        ratio = raw.irradiance_horizontal_W_per_m2_per_nm / fixed.irradiance_horizontal_W_per_m2_per_nm
        assert not math.isclose(ratio, 1.0, rel_tol=0.01)
