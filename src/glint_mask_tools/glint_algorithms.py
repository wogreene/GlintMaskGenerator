"""Module with classes that handle glint detection on the preprocessed pixel intensity values from sensor captures.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

EPSILON = 1e-8


class GlintAlgorithm(ABC):
    """Abstract base class that handles the glint detection logic on data from sensor captures."""

    def __init__(self) -> None:
        """Create a new glint masking algorithm instance."""
        super().__init__()

    @abstractmethod
    def __call__(
        self,
        img: np.ndarray,
        band_scales: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Return a boolean glint mask for a given image.

        Parameters
        ----------
        img
            Preprocessed multi-band image (H, W, C). For MicaSense sensors this
            is per-band surface reflectance (dimensionless, roughly 0-1 for
            diffuse, >1 for specular). For sensors without radiometric metadata
            it's ``DN / (2^bits - 1)``.
        band_scales
            Optional per-band scale factors (from ImageLoader.read_band_scales)
            for this capture. Only used by algorithms operating on raw DN space;
            algorithms working on reflectance can ignore it.

        Returns
        -------
        Output mask should have 1 for masked, 0 for unmasked.

        """
        raise NotImplementedError


class ThresholdAlgorithm(GlintAlgorithm):
    """Algorithm for estimating glint in an image using a simple disjunctive threshold on the band data values."""

    def __init__(self, thresholds: Sequence[float], *, per_band: bool = False) -> None:
        """Create a new ThresholdAlgorithm instance.

        Parameters
        ----------
        thresholds
            The threshold values for each band.
        per_band
            If True, return separate masks for each band. If False, combine
            all bands with logical OR (default behavior).

        """
        super().__init__()
        self.thresholds = thresholds
        self.per_band = per_band

    def __call__(
        self,
        img: np.ndarray,
        band_scales: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Apply the threshold masking algorithm to an img."""
        if self.per_band:
            return img > self.thresholds  # Returns (H, W, C) boolean
        return np.any(img > self.thresholds, axis=2)


class ChromaticityDiscriminatedThresholdAlgorithm(GlintAlgorithm):
    """Threshold algorithm that uses Red vs Blue chromaticity to reject shallow benthos.

    Sunlight glint and breaking-wave foam are broad-spectrum reflectors — their
    exposure-normalized reflectance in Red is close to (or slightly less than)
    their reflectance in Blue, giving a "redness" index near zero or slightly
    negative. Shallow live benthos (corals with pink/orange/yellow pigments,
    coralline algae, red-fluorescent species) absorbs Blue via chlorophyll and
    reflects or emits Red much more strongly, giving a distinctly positive
    redness. This is a more robust discriminator than NDVI for underwater
    scenes because water absorbs NIR aggressively and coral fluorescence
    complicates the NIR/Red ratio, whereas the Blue absorption / Red emission
    pattern survives shallow depths.

    Redness index:
      R = (Red − Blue) / (Red + Blue)   (both exposure-normalized)

    A pixel is marked as glint/foam if:
      it triggers any band's brightness threshold AND redness < redness_max

    ``band_scales`` (from ImageLoader.read_band_scales) is used to divide out
    per-band ExposureTime × ISO before computing redness, so the index reflects
    the physical spectral shape rather than DN bias from auto-exposure.
    """

    _EPSILON = 1e-8  # avoid divide-by-zero on deep-water pixels near black

    def __init__(  # noqa: PLR0913
        self,
        thresholds: Sequence[float],
        red_band_idx: int,
        blue_band_idx: int,
        redness_max: float,
        *,
        per_band: bool = False,
    ) -> None:
        """Create a new ChromaticityDiscriminatedThresholdAlgorithm.

        Parameters
        ----------
        thresholds
            Per-band brightness thresholds (same convention as ThresholdAlgorithm).
        red_band_idx
            Array index of the Red band used for the redness computation.
            For MicaSense sensors, Red 668 (Camera A) is the recommended choice.
        blue_band_idx
            Array index of the Blue band used for the redness computation.
            For MicaSense sensors, Blue 475 (Camera A) is the recommended choice.
            Using bands on the same physical camera avoids alignment residuals.
        redness_max
            Redness upper bound for a pixel to be considered glint/foam. Values
            below the threshold are eligible to be masked; values above are
            treated as spectrally-colored benthos and excluded. Empirically
            around 0.1 with exposure-normalized values; tune upward if bleached
            or pale coral slips through, downward if real whitewash is rejected.
        per_band
            If True, return per-band masks (same semantics as ThresholdAlgorithm).

        """
        super().__init__()
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.red_band_idx = red_band_idx
        self.blue_band_idx = blue_band_idx
        self.redness_max = redness_max
        self.per_band = per_band

    def __call__(
        self,
        img: np.ndarray,
        band_scales: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Apply the chromaticity-discriminated threshold to an image.

        Assumes ``img`` is per-band reflectance (or DN-normalized where
        auto-exposure is close to uniform across bands). Reflectance ratios are
        already properly scaled, so no exposure normalization is needed here.
        """
        red = img[:, :, self.red_band_idx]
        blue = img[:, :, self.blue_band_idx]
        redness = (red - blue) / (red + blue + self._EPSILON)
        is_flat_by_chroma = redness < self.redness_max  # (H, W) boolean

        per_band_hit = img > self.thresholds  # (H, W, C) boolean

        # The chromaticity test applies to every triggered pixel — a pixel is
        # masked only if it also looks spectrally flat (Red not much brighter
        # than Blue). This filters out shallow benthos regardless of which band
        # caught the brightness trigger.
        if self.per_band:
            return per_band_hit & is_flat_by_chroma[:, :, np.newaxis]
        return np.any(per_band_hit, axis=2) & is_flat_by_chroma


class IntensityRatioAlgorithm(GlintAlgorithm):
    """Class for estimating the specular reflection component of pixels in an image.

    Based on method from:
        Wang, S., Yu, C., Sun, Y. et al. Specular reflection removal
        of ocean surface remote sensing images from UAVs. Multimedia Tools
        Appl 77, 11363-11379 (2018). https://doi.org/10.1007/s11042-017-5551-7
    """

    def __init__(self, percent_diffuse: float = 0.95, threshold: float = 0.99) -> None:
        """Create and return a glint mask for RGB imagery.

        Parameters
        ----------
        percent_diffuse
            An estimate of the percentage of pixels in an image that show pure diffuse
            reflectance, and thus no specular reflectance (glint).
        threshold
            Threshold on specular reflectance estimate to binarize into a mask.
            e.g. if more than 50% specular reflectance is unacceptable, use 0.5.

        """
        super().__init__()
        self.percent_diffuse = percent_diffuse
        self.threshold = threshold

    def __call__(
        self,
        img: np.ndarray,
        band_scales: np.ndarray | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Create and return a glint mask for RGB imagery.

        Parameters
        ----------
        img: np.ndarray shape=(H,W,3)
            Path to a 3-channel RGB numpy image normalized to values in [0,1].
        band_scales
            Ignored; kept for interface consistency.

        Returns
        -------
        numpy.ndarray, shape=(H,W)
            Numpy array of glint mask for img at input_path.

        """
        return self._estimate_specular_reflection_component(img, self.percent_diffuse) > self.threshold

    @staticmethod
    def _estimate_specular_reflection_component(
        img: np.ndarray,
        percent_diffuse: float,
    ) -> np.ndarray:
        """Estimate the specular reflection component of pixels in an image.

        Based on method from:
            Wang, S., Yu, C., Sun, Y. et al. Specular reflection removal
            of ocean surface remote sensing images from UAVs. Multimedia Tools
            Appl 77, 11363-11379 (2018). https://doi.org/10.1007/s11042-017-5551-7

        Parameters
        ----------
        img: numpy.ndarray, shape=(H,W,C)
            A numpy ndarray of an RGB image.
        percent_diffuse
            An estimate of the % of pixels that show purely diffuse reflection.

        Returns
        -------
        numpy.ndarray, shape=(H,W)
            1D image with values being an estimate of specular reflectance.

        """
        # Calculate the pixel-wise max intensity and intensity range over RGB channels
        i_max = np.amax(img, axis=2)
        i_min = np.amin(img, axis=2)
        i_range = i_max - i_min

        # Calculate intensity ratio
        q = np.divide(i_max, i_range + EPSILON)

        # Select diffuse only pixels using the PERCENTILE_THRESH
        # i.e. A percentage of PERCENTILE_THRESH pixels are supposed to have no
        #     specular reflection
        num_thresh = math.ceil(percent_diffuse * q.size)

        # Get intensity ratio by separating the image into diffuse and specular sections
        q_x_hat = np.partition(q.ravel(), num_thresh)[num_thresh]

        # Estimate the spectral component of each pixel
        return np.clip(i_max - (q_x_hat * i_range), 0, None)
