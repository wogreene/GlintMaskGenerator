"""Tests for the water-column index that spares submerged benthos from the mask.

Created by: Taylor Denouden
Organization: Hakai Institute
"""

import numpy as np
import pytest

from glint_mask_tools.glint_algorithms import SurfaceDiscriminatedThresholdAlgorithm
from glint_mask_tools.sensors import m3m_sensor, msre_dual_sensor, msre_sensor, rgb_sensor

# Median index values measured over labelled pixels in Abaco reef imagery
# (IMG_0225 / IMG_0350), RedEdge717 vs NIR842.
GLINT_INDEX = 0.13
WHITEWASH_INDEX = 0.23
CORAL_INDEX = 0.57
REEF_INDEX = 0.60
CUT = 0.35


def bands_with_index(index, denominator=0.02):
    """Return (numerator, denominator) reflectances giving this normalized index."""
    return denominator * (1 + index) / (1 - index), denominator


class TestSensorConfiguration:
    """MicaSense sensors straddle water's near-infrared absorption rise."""

    @pytest.mark.parametrize("sensor", [msre_sensor, msre_dual_sensor])
    def test_micasense_uses_red_edge_over_nir(self, sensor):
        """Numerator is Red Edge 717, denominator is NIR 842 — both Camera A."""
        assert sensor.benthos_index_bands == (4, 3)
        numerator, denominator = sensor.benthos_index_bands
        assert sensor.bands[numerator].name.startswith("Red Edge")
        assert sensor.bands[denominator].name.startswith("Near-IR")

    def test_numerator_is_the_less_absorbed_band(self):
        """The pair only works with the deeper-penetrating band on top."""
        numerator, denominator = msre_dual_sensor.benthos_index_bands
        assert numerator == msre_dual_sensor.bands.index(
            next(b for b in msre_dual_sensor.bands if b.name.startswith("Red Edge 717"))
        )
        assert denominator == msre_dual_sensor.glint_reference_band  # NIR, the glint band

    def test_sensor_without_bands_rejects_the_option(self, tmp_path):
        """A sensor with no index pair can't use the discriminator."""
        with pytest.raises(ValueError, match="benthos_index_bands"):
            rgb_sensor.create_masker(
                str(tmp_path), str(tmp_path), rgb_sensor.get_default_thresholds(), 0, benthos_index_max=CUT
            )

    def test_multispectral_sensors_all_declare_a_pair(self):
        """Every sensor with a red edge and a NIR band can discriminate."""
        assert m3m_sensor.benthos_index_bands is not None

    def test_create_masker_wires_the_declared_bands(self, tmp_path):
        """The sensor's band pair reaches the algorithm."""
        masker = msre_dual_sensor.create_masker(
            str(tmp_path),
            str(tmp_path),
            msre_dual_sensor.get_default_thresholds(),
            0,
            benthos_index_max=CUT,
        )
        assert masker.algorithm.numerator_band_idx == 4
        assert masker.algorithm.denominator_band_idx == 3
        assert masker.algorithm.index_max == CUT


class TestDiscrimination:
    """Surface features stay masked, anything seen through water is spared."""

    @staticmethod
    def _algo(index_max=CUT):
        return SurfaceDiscriminatedThresholdAlgorithm(
            [0.01] * 5,
            numerator_band_idx=4,
            denominator_band_idx=3,
            index_max=index_max,
        )

    @staticmethod
    def _img(index):
        """A 5-band bright image whose water-column index is as given."""
        numerator, denominator = bands_with_index(index)
        img = np.full((4, 4, 5), 0.05, dtype=np.float32)
        img[:, :, 3] = denominator
        img[:, :, 4] = numerator
        return img

    @pytest.mark.parametrize("index", [GLINT_INDEX, WHITEWASH_INDEX])
    def test_surface_features_are_masked(self, index):
        """Glint and foam reflect off the surface, so little water absorbs their NIR."""
        assert self._algo()(self._img(index)).all()

    @pytest.mark.parametrize("index", [CORAL_INDEX, REEF_INDEX])
    def test_submerged_features_are_spared(self, index):
        """Reef and coral are seen through water, which suppresses NIR relative to 717."""
        assert not self._algo()(self._img(index)).any()

    def test_measured_populations_straddle_the_default_cut(self):
        """The default sits in the gap between the two measured populations."""
        assert WHITEWASH_INDEX < CUT < CORAL_INDEX

    def test_raising_the_cut_starts_masking_reef(self):
        """A cut above the reef index stops sparing it."""
        assert self._algo(index_max=0.7)(self._img(REEF_INDEX)).all()

    def test_brightness_is_still_required(self):
        """The index alone never masks anything; a brightness trigger is needed."""
        assert not self._algo()(self._img(GLINT_INDEX) * 0.01).any()
