"""Tests for the RGB whitewash detector.

Created by: Taylor Denouden
Organization: Hakai Institute
"""

import numpy as np
import pytest

from glint_mask_tools.glint_algorithms import WhitewashAlgorithm, local_background
from glint_mask_tools.sensors import msre_dual_sensor, rgb_sensor

# Median RGB values (0-1) measured on Abaco reef frames.
REEF = (0.43, 0.55, 0.47)
TURQUOISE_SAND = (0.15, 0.70, 0.70)
WHITEWASH = (0.98, 1.00, 0.99)
WARM_CLIPPED_FOAM = (1.00, 1.00, 0.97)  # foam that is a hair warm — must still be masked
MODERATE_FOAM = (0.70, 0.76, 0.72)
PALE_REEF = (0.66, 0.72, 0.70)
PALMATA = (0.52, 0.59, 0.46)


@pytest.fixture
def scene():
    """A 120x120 reef scene with one patch of each class, and their centre pixels."""
    img = np.empty((120, 120, 3), dtype=np.float32)
    img[:] = REEF
    patches = {
        "sand": ((0, 40), (0, 40), TURQUOISE_SAND),
        "whitewash": ((0, 40), (80, 120), WHITEWASH),
        "warm foam": ((45, 50), (45, 50), WARM_CLIPPED_FOAM),
        "moderate foam": ((60, 64), (20, 24), MODERATE_FOAM),
        "pale reef": ((70, 120), (70, 120), PALE_REEF),
        "palmata": ((100, 104), (10, 14), PALMATA),
    }
    centres = {}
    for name, ((y0, y1), (x0, x1), rgb) in patches.items():
        img[y0:y1, x0:x1] = rgb
        centres[name] = ((y0 + y1) // 2, (x0 + x1) // 2)
    return img, centres


class TestClasses:
    """Each measured class lands on the right side of the mask."""

    @pytest.mark.parametrize("name", ["whitewash", "warm foam", "moderate foam"])
    def test_surface_features_are_masked(self, scene, name):
        img, centres = scene
        assert WhitewashAlgorithm()(img)[centres[name]]

    @pytest.mark.parametrize("name", ["sand", "palmata", "pale reef"])
    def test_benthos_is_spared(self, scene, name):
        img, centres = scene
        assert not WhitewashAlgorithm()(img)[centres[name]]

    def test_plain_reef_is_spared(self, scene):
        img, _ = scene
        assert not WhitewashAlgorithm()(img)[110, 60]


class TestOrangeGuard:
    """Coloured orange pixels are exempt; warm foam is not."""

    # Bright and nearly colourless, but clearly orange: masked without the guard.
    BRIGHT_ORANGE = (0.99, 0.95, 0.84)

    def _img(self, rgb):
        img = np.full((60, 60, 3), REEF, dtype=np.float32)
        img[28:32, 28:32] = rgb
        return img

    def test_guard_spares_bright_orange(self):
        assert not WhitewashAlgorithm()(self._img(self.BRIGHT_ORANGE))[30, 30]

    def test_guard_can_be_turned_off(self):
        assert WhitewashAlgorithm(spare_orange=False)(self._img(self.BRIGHT_ORANGE))[30, 30]

    def test_warm_clipped_foam_is_not_treated_as_orange(self):
        """Red beats blue here, but the pixel is colourless — it's foam."""
        assert WhitewashAlgorithm()(self._img(WARM_CLIPPED_FOAM))[30, 30]


class TestBehaviour:
    """Plumbing and edge cases."""

    def test_saturation_flags_count_as_very_bright(self):
        """A pixel clipped in every channel is masked even if the scaled value is lower."""
        img = np.full((40, 40, 3), REEF, dtype=np.float32)
        img[20, 20] = (0.75, 0.75, 0.75)
        saturated = np.zeros(img.shape, dtype=bool)
        saturated[20, 20] = True
        # Contrast test disabled so only the clipping can trigger it.
        detector = WhitewashAlgorithm(local_contrast=100)
        assert not detector(img)[20, 20]
        assert detector(img, saturated=saturated)[20, 20]

    def test_per_band_repeats_the_mask(self, scene):
        img, _ = scene
        mask = WhitewashAlgorithm(per_band=True)(img)
        assert mask.shape == img.shape
        assert (mask == mask[:, :, :1]).all()

    def test_local_background_matches_input_shape_at_any_resolution(self):
        for shape in ((50, 70), (960, 1280), (1365, 2048)):
            assert local_background(np.zeros(shape, dtype=np.float32)).shape == shape


class TestSensorWiring:
    """Only plain RGB sensors offer the detector."""

    def test_rgb_builds_the_detector(self, tmp_path):
        masker = rgb_sensor.create_masker(
            str(tmp_path), str(tmp_path), rgb_sensor.get_default_thresholds(), 0, whitewash={"bright_floor": 0.9}
        )
        assert isinstance(masker.algorithm, WhitewashAlgorithm)
        assert masker.algorithm.bright_floor == 0.9

    def test_multispectral_rejects_it(self, tmp_path):
        with pytest.raises(ValueError, match="whitewash"):
            msre_dual_sensor.create_masker(
                str(tmp_path), str(tmp_path), msre_dual_sensor.get_default_thresholds(), 0, whitewash={}
            )
