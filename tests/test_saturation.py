"""Tests for saturation-aware masking and threshold reachability.

Created by: Taylor Denouden
Organization: Hakai Institute
"""

import dataclasses
import math

import numpy as np
from loguru import logger

from glint_mask_tools.glint_algorithms import (
    SurfaceDiscriminatedThresholdAlgorithm,
    ThresholdAlgorithm,
    contrast_hits,
)
from glint_mask_tools.irradiance import IrradianceCalibrator
from glint_mask_tools.radiometric import SATURATION_DN_FRACTION, saturation_reflectance_ceiling
from glint_mask_tools.sensors import msre_dual_sensor, rgb_sensor
from tests.test_irradiance import GOOD_ATTITUDE, make_capture, make_meta


class TestThresholdAlgorithmSaturation:
    """Clipped pixels are detections even when their converted value is low."""

    @staticmethod
    def _img():
        """A (4, 4, 2) reflectance image whose values are all well below threshold."""
        return np.full((4, 4, 2), 0.001, dtype=np.float32)

    def test_low_value_pixels_are_not_masked_without_saturation_flags(self):
        """Baseline: nothing reaches the threshold, so nothing is masked."""
        algo = ThresholdAlgorithm([0.03, 0.03])
        assert not algo(self._img()).any()

    def test_saturated_pixel_is_masked_despite_low_value(self):
        """A clipped pixel is masked even though its reflectance is under threshold."""
        saturated = np.zeros((4, 4, 2), dtype=bool)
        saturated[1, 2, 0] = True
        mask = ThresholdAlgorithm([0.03, 0.03])(self._img(), saturated=saturated)
        assert mask[1, 2]
        assert mask.sum() == 1

    def test_saturation_unions_with_threshold_hits(self):
        """Threshold detections survive alongside the saturation flags."""
        img = self._img()
        img[0, 0, 1] = 0.5
        saturated = np.zeros((4, 4, 2), dtype=bool)
        saturated[3, 3, 0] = True
        mask = ThresholdAlgorithm([0.03, 0.03])(img, saturated=saturated)
        assert mask[0, 0]
        assert mask[3, 3]
        assert mask.sum() == 2

    def test_per_band_saturation_is_band_specific(self):
        """In per-band mode a clipped band only marks that band."""
        saturated = np.zeros((4, 4, 2), dtype=bool)
        saturated[2, 2, 1] = True
        mask = ThresholdAlgorithm([0.03, 0.03], per_band=True)(self._img(), saturated=saturated)
        assert mask.shape == (4, 4, 2)
        assert mask[2, 2, 1]
        assert not mask[2, 2, 0]


class TestChromaticitySaturation:
    """Saturation feeds the brightness test but still passes through the surface gate."""

    @staticmethod
    def _algo():
        return SurfaceDiscriminatedThresholdAlgorithm(
            [0.03] * 3,
            numerator_band_idx=0,
            denominator_band_idx=2,
            index_max=0.1,
        )

    def test_spectrally_flat_saturated_pixel_is_masked(self):
        """Clipped glint (flat spectrum) is masked despite a low converted value."""
        img = np.full((3, 3, 3), 0.001, dtype=np.float32)
        saturated = np.zeros((3, 3, 3), dtype=bool)
        saturated[1, 1, 0] = True
        assert self._algo()(img, saturated=saturated)[1, 1]

    def test_red_saturated_pixel_is_still_rejected_as_benthos(self):
        """The surface gate applies to clipped pixels too."""
        img = np.full((3, 3, 3), 0.001, dtype=np.float32)
        img[1, 1, 0] = 0.5  # numerator band far above denominator -> index above the bound
        saturated = np.zeros((3, 3, 3), dtype=bool)
        saturated[1, 1, 0] = True
        assert not self._algo()(img, saturated=saturated)[1, 1]


class TestSaturationDn:
    """Per-sensor clipping level."""

    def test_eight_bit_sensor(self):
        """An 8-bit sensor clips just below 255."""
        assert rgb_sensor.saturation_dn == SATURATION_DN_FRACTION * 255

    def test_sixteen_bit_sensor_catches_micasense_ceiling(self):
        """MicaSense writes 12-bit data shifted into 16 bits, topping out at 65520."""
        assert msre_dual_sensor.saturation_dn < 65520


class TestSaturationReflectanceCeiling:
    """The maximum reflectance a capture can express."""

    def test_ceiling_scales_inversely_with_exposure(self):
        """Doubling exposure halves the highest expressible reflectance."""
        short = make_meta(**GOOD_ATTITUDE)
        long = dataclasses.replace(short, exposure_time_s=short.exposure_time_s * 2)
        assert math.isclose(
            saturation_reflectance_ceiling(short) / 2,
            saturation_reflectance_ceiling(long),
            rel_tol=1e-6,
        )

    def test_ceiling_scales_inversely_with_gain(self):
        """Higher ISO also lowers the ceiling."""
        low = make_meta(**GOOD_ATTITUDE)
        high = dataclasses.replace(low, iso=low.iso * 8)
        assert math.isclose(
            saturation_reflectance_ceiling(low) / 8,
            saturation_reflectance_ceiling(high),
            rel_tol=1e-6,
        )

    def test_long_exposure_puts_threshold_out_of_reach(self):
        """The real failure mode: a long exposure drops the ceiling under the threshold."""
        # Values from the Abaco flight: ~4 ms at ISO 800 put the NIR ceiling at
        # ~0.02, below a 0.03 threshold, so no pixel could ever trigger it.
        meta = dataclasses.replace(
            make_meta(irradiance=40.0, **GOOD_ATTITUDE),
            exposure_time_s=0.004,
            iso=800,
            radiometric_calibration=(0.00013, 0.0, 0.0),
        )
        assert saturation_reflectance_ceiling(meta) < 0.03

    def test_non_positive_irradiance_is_unbounded(self):
        """A malformed DLS reading can't constrain the ceiling."""
        meta = dataclasses.replace(make_meta(), horizontal_irradiance_override=0.0)
        assert saturation_reflectance_ceiling(meta) == float("inf")


class TestUnreachableThresholdWarning:
    """The run should say so when a threshold can't be reached."""

    @staticmethod
    def _calibrated(exposure_time_s):
        captures = [
            [
                dataclasses.replace(
                    m,
                    exposure_time_s=exposure_time_s,
                    iso=800,
                    irradiance_micro_W_per_cm2_per_nm=40.0,
                    radiometric_calibration=(0.00013, 0.0, 0.0),
                )
                for m in make_capture(i, reliable=True, n_bands=2)
            ]
            for i in range(10)
        ]
        calibrator = IrradianceCalibrator()
        calibrator.calibrate(list(range(len(captures))), lambda k: captures[k])
        return calibrator

    @staticmethod
    def _capture_logs(fn):
        messages = []
        sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
        try:
            fn()
        finally:
            logger.remove(sink_id)
        return messages

    def test_warns_when_ceiling_is_below_threshold(self):
        """A long exposure with a high threshold produces a warning."""
        calibrator = self._calibrated(0.004)
        messages = self._capture_logs(lambda: calibrator.warn_on_unreachable_thresholds([0.03, 0.03]))
        assert any("unreachable" in m for m in messages)

    def test_silent_when_threshold_is_reachable(self):
        """A short exposure leaves plenty of headroom, so nothing is logged."""
        calibrator = self._calibrated(0.0005)
        messages = self._capture_logs(lambda: calibrator.warn_on_unreachable_thresholds([0.03, 0.03]))
        assert not messages

    def test_ignores_threshold_count_mismatch(self):
        """A threshold list that doesn't match the band count is skipped, not crashed."""
        calibrator = self._calibrated(0.004)
        messages = self._capture_logs(lambda: calibrator.warn_on_unreachable_thresholds([0.03] * 5))
        assert not messages

    def test_uncalibrated_calibrator_does_not_warn(self):
        """Nothing to report before a sweep has run."""
        messages = self._capture_logs(lambda: IrradianceCalibrator().warn_on_unreachable_thresholds([0.03]))
        assert not messages


class TestContrastHits:
    """Scene-relative detection, for captures whose reflectance scale is compressed."""

    @staticmethod
    def _img(bright_value=1.0, background=0.01):
        """A (10, 10, 2) image with a uniform background and one bright pixel."""
        img = np.full((10, 10, 2), background, dtype=np.float32)
        img[4, 4, :] = bright_value
        return img

    def test_flags_pixels_above_the_background_multiple(self):
        """A pixel far above the frame's own background is flagged."""
        hits = contrast_hits(self._img(), 3.0, reference_band=0)
        assert hits.shape == (10, 10, 1)
        assert hits[4, 4, 0]
        assert hits.sum() == 1

    def test_ignores_pixels_below_the_multiple(self):
        """A pixel only slightly above background stays unflagged."""
        assert not contrast_hits(self._img(bright_value=0.02), 3.0, reference_band=0).any()

    def test_is_invariant_to_overall_scaling(self):
        """Halving every value (as a longer exposure would) changes nothing."""
        img = self._img()
        assert np.array_equal(
            contrast_hits(img, 3.0, reference_band=0),
            contrast_hits(img / 2.0, 3.0, reference_band=0),
        )

    def test_reference_band_is_respected(self):
        """Only the reference band's contrast is consulted."""
        img = self._img()
        img[4, 4, 0] = 0.01  # bright in band 1 only
        assert not contrast_hits(img, 3.0, reference_band=0).any()
        assert contrast_hits(img, 3.0, reference_band=1)[4, 4, 0]

    def test_without_reference_band_any_band_counts(self):
        """With no reference band, a hit in any band flags the pixel."""
        img = self._img()
        img[4, 4, 0] = 0.01
        assert contrast_hits(img, 3.0)[4, 4, 0]

    def test_zero_background_flags_nothing(self):
        """A non-positive background would make the multiple meaningless."""
        img = np.zeros((6, 6, 2), dtype=np.float32)
        img[0, 0, :] = 1.0
        assert not contrast_hits(img, 3.0, reference_band=0).any()
        assert not contrast_hits(img, 3.0).any()


class TestThresholdAlgorithmContrast:
    """The contrast rule as wired into the threshold algorithm."""

    @staticmethod
    def _img():
        """Clipped-capture stand-in: bright glint that still falls below threshold."""
        img = np.full((10, 10, 2), 0.002, dtype=np.float32)
        img[4, 4, :] = 0.02  # 10x the background, but under a 0.03 threshold
        return img

    def test_disabled_by_default(self):
        """Without a multiplier the algorithm is unchanged."""
        assert not ThresholdAlgorithm([0.03, 0.03])(self._img()).any()

    def test_contrast_rule_recovers_sub_threshold_glint(self):
        """With the multiplier set, the bright pixel is masked."""
        algo = ThresholdAlgorithm([0.03, 0.03], contrast_multiplier=3.0, reference_band=0)
        assert algo(self._img())[4, 4]

    def test_unions_with_absolute_threshold(self):
        """Absolute detections elsewhere in the frame still stand."""
        img = self._img()
        img[0, 0, 1] = 0.5
        mask = ThresholdAlgorithm([0.03, 0.03], contrast_multiplier=3.0, reference_band=0)(img)
        assert mask[0, 0]
        assert mask[4, 4]

    def test_chromaticity_algorithm_gates_contrast_hits(self):
        """Contrast detections still have to look like surface features."""
        img = np.full((10, 10, 3), 0.002, dtype=np.float32)
        img[4, 4, :] = 0.02
        flat = SurfaceDiscriminatedThresholdAlgorithm(
            [0.03] * 3, numerator_band_idx=0, denominator_band_idx=2, index_max=0.1,
            contrast_multiplier=3.0, reference_band=1,
        )
        assert flat(img)[4, 4]

        img[4, 4, 0] = 0.06  # much redder than blue -> benthos, not glint
        assert not flat(img)[4, 4]


class TestSensorGlintReferenceBand:
    """Sensors with a NIR band designate it for contrast detection."""

    def test_micasense_dual_uses_nir(self):
        """NIR 842 sits at index 3 for the RedEdge-MX Dual."""
        assert msre_dual_sensor.glint_reference_band == 3
        assert msre_dual_sensor.bands[3].name.startswith("Near-IR")

    def test_rgb_has_no_reference_band(self):
        """RGB has no NIR, so contrast falls back to any-band."""
        assert rgb_sensor.glint_reference_band is None
