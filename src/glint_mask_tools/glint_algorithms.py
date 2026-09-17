"""Module with classes that handle glint detection on the preprocessed pixel intensity values from sensor captures.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

EPSILON = 1e-8


def contrast_hits(
    img: np.ndarray,
    multiplier: float,
    reference_band: int | None = None,
) -> np.ndarray:
    """Flag pixels that stand out against their own capture's background.

    Thresholding calibrated reflectance assumes the numbers are comparable
    between captures. They stop being comparable the moment glint clips the
    sensor: everything above the well depth converts to the same value, and
    that value moves with whatever exposure the camera chose, so a fixed
    threshold can sit above anything a capture is able to express. Contrast
    against the frame's own water background survives all of that — it only
    cares whether a pixel is brighter than its surroundings, which is what
    glint is — at the cost of always finding *something*, even in a frame with
    no real glint in it.

    Parameters
    ----------
    img
        Preprocessed (H, W, C) image.
    multiplier
        How many times the background level a pixel must reach. Around 3 tracks
        visible glint on MicaSense water imagery; lower catches broad sheen,
        higher isolates hard specular highlights only.
    reference_band
        Band index to judge contrast in. Glint is brightest in NIR and water is
        darkest there, so the sensor's NIR band gives the cleanest separation.
        With None, a pixel counts if *any* band exceeds its own background,
        which on a 10-band sensor masks far more than NIR alone.

    Returns
    -------
    (H, W, 1) boolean array, ready to broadcast across bands.

    """
    if reference_band is not None:
        band = img[:, :, reference_band]
        background = float(np.median(band))
        if background <= 0:
            return np.zeros((*img.shape[:2], 1), dtype=bool)
        return (band > multiplier * background)[:, :, np.newaxis]

    background = np.median(img, axis=(0, 1))
    usable = background > 0
    if not usable.any():
        return np.zeros((*img.shape[:2], 1), dtype=bool)
    hits = (img[:, :, usable] > multiplier * background[usable]).any(axis=2)
    return hits[:, :, np.newaxis]


class GlintAlgorithm(ABC):
    """Abstract base class that handles the glint detection logic on data from sensor captures."""

    def __init__(self) -> None:
        """Create a new glint masking algorithm instance."""
        super().__init__()

    @abstractmethod
    def __call__(
        self,
        img: np.ndarray,
        band_scales: np.ndarray | None = None,
        saturated: np.ndarray | None = None,
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
        saturated
            Optional (H, W, C) boolean array flagging pixels that hit the
            sensor's full well in the raw data. Their converted value is a floor,
            not a measurement — radiance divides by exposure time, so a clipped
            pixel from a long exposure converts to a *low* reflectance and can
            land below the threshold however bright the scene really was.
            Algorithms treat these as detections regardless of threshold.

        Returns
        -------
        Output mask should have 1 for masked, 0 for unmasked.

        """
        raise NotImplementedError


class ThresholdAlgorithm(GlintAlgorithm):
    """Algorithm for estimating glint in an image using a simple disjunctive threshold on the band data values."""

    def __init__(
        self,
        thresholds: Sequence[float],
        *,
        per_band: bool = False,
        contrast_multiplier: float | None = None,
        reference_band: int | None = None,
    ) -> None:
        """Create a new ThresholdAlgorithm instance.

        Parameters
        ----------
        thresholds
            The threshold values for each band.
        per_band
            If True, return separate masks for each band. If False, combine
            all bands with logical OR (default behavior).
        contrast_multiplier
            If set, also flag pixels exceeding this multiple of their capture's
            own background level (see ``contrast_hits``). Unioned with the
            absolute thresholds, so it adds detections on captures whose
            reflectance scale is compressed by clipping without changing
            captures where the thresholds already work.
        reference_band
            Band the contrast rule is judged in; see ``contrast_hits``.

        """
        super().__init__()
        self.thresholds = thresholds
        self.per_band = per_band
        self.contrast_multiplier = contrast_multiplier
        self.reference_band = reference_band

    def __call__(
        self,
        img: np.ndarray,
        band_scales: np.ndarray | None = None,  # noqa: ARG002
        saturated: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the threshold masking algorithm to an img."""
        flagged = img > self.thresholds  # (H, W, C) boolean
        if self.contrast_multiplier is not None:
            flagged = flagged | contrast_hits(img, self.contrast_multiplier, self.reference_band)
        if saturated is not None:
            flagged = flagged | saturated
        if self.per_band:
            return flagged
        return np.any(flagged, axis=2)


class SurfaceDiscriminatedThresholdAlgorithm(GlintAlgorithm):
    """Threshold algorithm that separates surface features from anything underwater.

    The useful question isn't "is this coral?" but "is there water above this
    pixel?". Glint and breaking-wave foam sit *on* the surface; reef, sand and
    coral sit *under* it. Water gives that away, because its absorption climbs
    steeply through the near infrared — roughly 1.3 /m at 717 nm against 4.8 /m
    at 842 nm. Even a hand's depth of water therefore suppresses 842 far more
    than 717, while a surface reflection passes through no water at all and
    returns both bands in the proportions the sky sent them.

    Water-column index:
      I = (numerator - denominator) / (numerator + denominator)

    with the *less* absorbed band on top. A pixel is masked if it triggers a
    brightness detection AND ``I < index_max``: low index means little or no
    water above it (glint, foam), high index means it is being seen through
    water (benthos) and is spared.

    Measured on Abaco reef imagery over labelled glint, whitewash, coral, reef
    and sand pixels:

      RedEdge717 vs NIR842 : 93% of glint/foam kept, 99% of reef spared (cut +0.35)
      RedEdge717 vs Red668 : 79% / 92% (cut +0.10) — a chlorophyll red-edge test,
                             so it reads live benthos but misses bare sand and rubble
      Red668 vs Blue475    : 63% / 87% (cut -0.19) — everything underwater is
                             blue-shifted, so this mostly measures depth; with the
                             +0.1 default intended for targets in air it rejects nothing

    Class medians for the RedEdge717/NIR842 pair: glint +0.13, whitewash +0.23,
    coral +0.57, reef +0.60, sand +0.57 — the gap is wide enough that the exact
    cut barely matters. Prefer two bands on the same physical camera so rig
    alignment keeps them sub-pixel registered.

    What this cannot do: separate faint glint *lying on top of* shallow reef.
    Light from such a pixel is mostly the reef's, so it reads as submerged.
    """

    _EPSILON = 1e-8  # avoid divide-by-zero on deep-water pixels near black

    def __init__(  # noqa: PLR0913
        self,
        thresholds: Sequence[float],
        numerator_band_idx: int,
        denominator_band_idx: int,
        index_max: float,
        *,
        per_band: bool = False,
        contrast_multiplier: float | None = None,
        reference_band: int | None = None,
    ) -> None:
        """Create a new ChromaticityDiscriminatedThresholdAlgorithm.

        Parameters
        ----------
        thresholds
            Per-band brightness thresholds (same convention as ThresholdAlgorithm).
        numerator_band_idx
            Array index of the *less* water-absorbed band. For MicaSense
            sensors, Red Edge 717 (Camera A).
        denominator_band_idx
            Array index of the *more* water-absorbed band. For MicaSense
            sensors, NIR 842 (Camera A) — same physical camera as the numerator,
            so rig alignment keeps the two sub-pixel registered.
        index_max
            Upper bound for a pixel to count as a surface feature. Below it a
            pixel is eligible to be masked; above it the pixel is being seen
            through water and is spared. Around +0.35 for the RedEdge717/NIR842
            pair; lower it to protect shallow benthos harder, raise it to catch
            more glint.
        per_band
            If True, return per-band masks (same semantics as ThresholdAlgorithm).
        contrast_multiplier, reference_band
            Same meaning as on ThresholdAlgorithm; contrast detections still
            have to clear the chromaticity test.

        """
        super().__init__()
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.numerator_band_idx = numerator_band_idx
        self.denominator_band_idx = denominator_band_idx
        self.index_max = index_max
        self.per_band = per_band
        self.contrast_multiplier = contrast_multiplier
        self.reference_band = reference_band

    def __call__(
        self,
        img: np.ndarray,
        band_scales: np.ndarray | None = None,  # noqa: ARG002
        saturated: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the spectrally-discriminated threshold to an image.

        Assumes ``img`` is per-band reflectance (or DN-normalized where
        auto-exposure is close to uniform across bands). Reflectance ratios are
        already properly scaled, so no exposure normalization is needed here.
        """
        numerator = img[:, :, self.numerator_band_idx]
        denominator = img[:, :, self.denominator_band_idx]
        index = (numerator - denominator) / (numerator + denominator + self._EPSILON)
        is_flat_spectrum = index < self.index_max  # (H, W) boolean

        per_band_hit = img > self.thresholds  # (H, W, C) boolean
        if self.contrast_multiplier is not None:
            per_band_hit = per_band_hit | contrast_hits(img, self.contrast_multiplier, self.reference_band)
        if saturated is not None:
            # Saturated pixels count as brightness hits, but still have to pass
            # the chromaticity test: clipping biases redness toward zero, so
            # letting them bypass it would mask saturated benthos as glint.
            per_band_hit = per_band_hit | saturated

        # The spectral test applies to every triggered pixel — a pixel is masked
        # only if it also looks spectrally flat. This spares live benthos
        # regardless of which band caught the brightness trigger.
        if self.per_band:
            return per_band_hit & is_flat_spectrum[:, :, np.newaxis]
        return np.any(per_band_hit, axis=2) & is_flat_spectrum


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
        saturated: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create and return a glint mask for RGB imagery.

        Parameters
        ----------
        img: np.ndarray shape=(H,W,3)
            Path to a 3-channel RGB numpy image normalized to values in [0,1].
        band_scales
            Ignored; kept for interface consistency.
        saturated
            Optional (H,W,C) boolean array of clipped pixels, unioned into the
            result: the specular estimate saturates along with the data.

        Returns
        -------
        numpy.ndarray, shape=(H,W)
            Numpy array of glint mask for img at input_path.

        """
        mask = self._estimate_specular_reflection_component(img, self.percent_diffuse) > self.threshold
        if saturated is not None:
            mask = mask | np.any(saturated, axis=2)
        return mask

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
