"""Sensor configuration module for dynamic generation of CLI commands and GUI interfaces.

This module defines sensor configurations that specify the bands and image loaders
for different camera _known_sensors. The configurations are used by both the CLI and GUI
to dynamically generate appropriate interfaces for each sensor type.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .glint_algorithms import ChromaticityDiscriminatedThresholdAlgorithm, ThresholdAlgorithm
from .image_loaders import (
    BigTiffLoader,
    DJIM3MLoader,
    ImageLoader,
    MicasenseRedEdgeDualLoader,
    MicasenseRedEdgeLoader,
    P4MSLoader,
    SingleFileImageLoader,
)
from .maskers import Masker
from .utils import normalize_img

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Band:
    """A sensor band with a name and default threshold value."""

    name: str
    default_threshold: float = 1.000


B = Band("Blue", 0.875)
G = Band("Green")
R = Band("Red")
RE = Band("Red Edge")
NIR = Band("Near-IR")


@dataclass
class Sensor:
    """Sensor configuration class that specifies the name and band order, as well as loader class to handle imagery."""

    name: str
    bands: list[Band]
    bit_depth: int
    loader_class: type[ImageLoader]
    # Upper bound of the useful threshold range for this sensor, used by the GUI to
    # scale the threshold sliders. Sensors that only bit-depth normalize their DN work
    # on a full [0, 1] scale, so they need the whole range. Sensors converted to
    # surface reflectance (MicaSense) live on a much smaller scale — real glint tops
    # out near 0.08 — so a 0-1 slider would waste almost all its travel.
    threshold_max: float = 1.0
    supports_alignment: bool = False
    # Index of the band to use as the alignment reference. Pick a band with strong texture
    # and high SNR over the expected scene content. Only used when supports_alignment=True.
    alignment_reference_band: int = 0
    # Default alignment strategy when align_bands is True. "rig" uses per-band XMP
    # calibration (only available for MicaSense currently); "phase" uses content-based
    # phase correlation; can be overridden at the call site.
    alignment_strategy: str = "phase"
    # Optional per-band physical-camera assignment. Bands on the same physical PCB
    # share the same integer id. Used by rig alignment to fit a single shared
    # rotation across bands on a foreign camera relative to the reference.
    # For example, MicaSense RedEdge-MX Dual has two PCBs → [0]*5 + [1]*5.
    bands_by_camera: list[int] | None = None
    # Band indices for Red-Blue chromaticity discrimination. When both are set,
    # callers can pass ``redness_max`` to create_masker to enable the chromaticity
    # discriminator that rejects shallow benthos (corals with colored pigments)
    # from the mask. Recommended: Red 668 (Camera A) + Blue 475 (Camera A) on
    # MicaSense sensors so both bands share a physical camera (perfect alignment).
    red_band_idx: int | None = None
    blue_band_idx: int | None = None

    def preprocess_image(
        self,
        img: np.ndarray,
        radiometric_metadata=None,  # noqa: ANN001
    ) -> np.ndarray:
        """Scale raw DN to a physically meaningful [0, ~few] range.

        When ``radiometric_metadata`` is supplied (MicaSense), applies the full
        MicaSense radiometric conversion → surface reflectance (dimensionless,
        typically 0-1 for diffuse and >1 for specular). Otherwise falls back to
        simple bit-depth normalization.
        """
        if radiometric_metadata is not None:
            from .radiometric import apply_reflectance_conversion  # noqa: PLC0415
            return apply_reflectance_conversion(img, radiometric_metadata)
        return normalize_img(img, bit_depth=self.bit_depth)

    def create_masker(  # noqa: PLR0913
        self,
        img_dir: str,
        mask_dir: str,
        thresholds: list[float],
        pixel_buffer: int,
        *,
        per_band: bool = False,
        align_bands: bool = True,
        alignment_strategy: str | None = None,
        redness_max: float | None = None,
        stabilize_irradiance: bool = True,
    ) -> Masker:
        """Create a masker instance for this sensor configuration.

        Parameters
        ----------
        alignment_strategy
            Override the sensor's default alignment strategy. One of "rig", "phase",
            or "none". If None, uses ``self.alignment_strategy``.
        redness_max
            If set (and the sensor has ``red_band_idx`` and ``blue_band_idx``
            configured), use the chromaticity-discriminated threshold algorithm.
            Pixels only stay in the mask if their redness index
            ``(Red − Blue) / (Red + Blue)`` (exposure-normalized) is below this
            bound. Rejects shallow colored benthos (coral, coralline algae)
            while preserving spectrally-flat glint and whitewash.
        stabilize_irradiance
            Build a flight-level DLS irradiance model and use it for captures
            whose own sun-sensor geometry is unusable (see the ``irradiance``
            module). Only affects sensors that convert DN to reflectance;
            harmless no-op elsewhere. Disable to get each capture's raw
            per-capture estimate.

        """
        strategy = alignment_strategy if alignment_strategy is not None else self.alignment_strategy
        aligner = None
        if self.supports_alignment and align_bands and strategy != "none":
            from .band_alignment import BandAligner, RigCalibratedAligner  # noqa: PLC0415

            if strategy == "rig":
                foreign_bands: list[int] | None = None
                if self.bands_by_camera is not None:
                    ref_cam = self.bands_by_camera[self.alignment_reference_band]
                    foreign_bands = [
                        i for i, c in enumerate(self.bands_by_camera) if c != ref_cam
                    ]
                aligner = RigCalibratedAligner(
                    reference_band=self.alignment_reference_band,
                    foreign_camera_bands=foreign_bands,
                    refinement_method="affine" if foreign_bands else "translation",
                )
            elif strategy == "phase":
                aligner = BandAligner(enabled=True, reference_band=self.alignment_reference_band)
            else:
                msg = f"Unknown alignment_strategy: {strategy!r}. Use 'rig', 'phase', or 'none'."
                raise ValueError(msg)

        if redness_max is not None and self.red_band_idx is not None and self.blue_band_idx is not None:
            algorithm = ChromaticityDiscriminatedThresholdAlgorithm(
                thresholds,
                red_band_idx=self.red_band_idx,
                blue_band_idx=self.blue_band_idx,
                redness_max=redness_max,
                per_band=per_band,
            )
        elif redness_max is not None:
            msg = (
                f"Sensor {self.name!r} doesn't declare red_band_idx / blue_band_idx; "
                "redness_max cannot be used."
            )
            raise ValueError(msg)
        else:
            algorithm = ThresholdAlgorithm(thresholds, per_band=per_band)

        from .irradiance import IrradianceCalibrator  # noqa: PLC0415

        return Masker(
            algorithm=algorithm,
            image_loader=self.loader_class(img_dir, mask_dir),
            image_preprocessor=self.preprocess_image,
            pixel_buffer=pixel_buffer,
            per_band=per_band,
            band_aligner=aligner,
            irradiance_calibrator=IrradianceCalibrator(enabled=stabilize_irradiance),
        )

    def get_default_thresholds(self) -> list[float]:
        """Get the default threshold values for all bands."""
        return [band.default_threshold for band in self.bands]


rgb_sensor = Sensor(
    name="RGB",
    bands=[R, G, B],
    bit_depth=8,
    loader_class=SingleFileImageLoader,
)
cir_sensor = Sensor(
    name="PhaseOne 4-band CIR",
    bands=[R, G, B, NIR],
    bit_depth=8,
    loader_class=BigTiffLoader,
    red_band_idx=0,
    blue_band_idx=2,
)
p4ms_sensor = Sensor(
    name="DJI P4MS",
    bands=[B, G, R, RE, NIR],
    bit_depth=16,
    loader_class=P4MSLoader,
    supports_alignment=True,
    red_band_idx=2,
    blue_band_idx=0,
)
# DJI M3M lacks a Blue band, so chromaticity discrimination isn't wired up here.
m3m_sensor = Sensor(
    name="DJI M3M",
    bands=[Band("Green", 0.875), R, RE, NIR],
    bit_depth=16,
    loader_class=DJIM3MLoader,
    supports_alignment=True,
)
# Band order below MUST match the file numbering produced by the camera, because the
# image loader stacks files _1.._N into array indices 0..N-1. MicaSense file numbering
# for the original RedEdge-MX is: _1=Blue, _2=Green, _3=Red, _4=NIR, _5=RedEdge.
# (Confirmed from per-file XMP BandName / CentralWavelength tags.)
msre_sensor = Sensor(
    name="MicaSense RedEdge",
    # This sensor's DN is converted to surface reflectance (not just bit-depth
    # normalized DN/65535), so thresholds live on a much smaller numeric scale
    # than the shared B/G/R/RE/NIR constants above (those are tuned for plain
    # DN normalization, used by p4ms/m3m/cir). Verified empirically: even
    # strong glint/wave-crest reflectance tops out around 0.05-0.08 on typical
    # MicaSense radiometric calibration data — nowhere near the "1.0 = 100%
    # reflectance" scale the numbers might suggest. 0.04 default catches real
    # glint across both favorable and oblique-attitude flight passes without
    # being wide open. Only NIR was rigorously verified this way; other bands
    # are set by analogy (glint is spectrally ~flat) and may need retuning.
    bands=[
        Band("Blue", 0.035),
        Band("Green", 0.04),
        Band("Red", 0.04),
        Band("Near-IR", 0.04),
        Band("Red Edge", 0.04),
    ],
    bit_depth=16,
    loader_class=MicasenseRedEdgeLoader,
    # Reflectance scale, not DN — see threshold_max docs on Sensor.
    threshold_max=0.5,
    supports_alignment=True,
    # NIR (index 3) is where glint is most visible — making it the reference means the
    # mask is computed in NIR's coordinate frame, so glint detection is geometrically
    # exact for the band that matters most.
    alignment_reference_band=3,
    alignment_strategy="rig",
    # Red 668 at index 2 (file _3) and Blue 475 at index 0 (file _1) — both on
    # Camera A, so they're sub-pixel aligned via the rig homography. Used for
    # the chromaticity discriminator that rejects shallow colored benthos.
    red_band_idx=2,
    blue_band_idx=0,
)

# RedEdge-MX Dual band order from XMP BandName/CentralWavelength tags. The first 5
# files are the original Camera A (RedEdge-MX) ordering; files _6.._10 are Camera B.
msre_dual_sensor = Sensor(
    name="MicaSense RedEdge-MX Dual",
    # Reflectance-scale thresholds (see msre_sensor comment above) — this
    # sensor also goes through full DN-to-reflectance conversion. NIR (index 3)
    # was the band rigorously verified against real flight data this session;
    # other bands set by analogy since glint is spectrally ~flat.
    bands=[
        Band("Blue 475(32)", 0.035),         # index 0 = file _1
        Band("Green 560(27)", 0.04),          # index 1 = file _2
        Band("Red 668(14)", 0.04),            # index 2 = file _3
        Band("Near-IR 842(57)", 0.04),        # index 3 = file _4
        Band("Red Edge 717(12)", 0.04),       # index 4 = file _5
        Band("Coastal Blue 444(28)", 0.035),  # index 5 = file _6
        Band("Green 531(14)", 0.04),          # index 6 = file _7
        Band("Red 650(16)", 0.04),            # index 7 = file _8
        Band("Red Edge 705(10)", 0.04),       # index 8 = file _9
        Band("Red Edge 740(18)", 0.04),       # index 9 = file _10
    ],
    bit_depth=16,
    loader_class=MicasenseRedEdgeDualLoader,
    # Reflectance scale, not DN — see threshold_max docs on Sensor.
    threshold_max=0.5,
    supports_alignment=True,
    # NIR (file _4, index 3): glint is most visible there, and it's a Camera A band,
    # so other Camera A bands get sub-pixel residuals from the rig homography.
    alignment_reference_band=3,
    alignment_strategy="rig",
    # First 5 files are Camera A (RedEdge-MX), last 5 are Camera B (RedEdge-MX Blue).
    # This lets the aligner fit a single shared rotation for all Camera B bands.
    bands_by_camera=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    # Red 668 at index 2 (file _3) and Blue 475 at index 0 (file _1). Both on
    # Camera A, so they're sub-pixel aligned via the rig homography. Used by
    # the chromaticity discriminator to distinguish spectrally-flat glint/foam
    # (redness ≈ 0) from colored benthos (redness > 0).
    red_band_idx=2,
    blue_band_idx=0,
)


# Auto populate GUI and CLI options
@dataclass(frozen=True)
class _KnownSensor:
    sensor: Sensor
    cli_name: str


_known_sensors = (
    _KnownSensor(rgb_sensor, cli_name="rgb"),
    _KnownSensor(cir_sensor, cli_name="cir"),
    _KnownSensor(p4ms_sensor, cli_name="p4ms"),
    _KnownSensor(m3m_sensor, cli_name="m3m"),
    _KnownSensor(msre_sensor, cli_name="msre"),
    _KnownSensor(msre_dual_sensor, cli_name="msre-dual"),
)
