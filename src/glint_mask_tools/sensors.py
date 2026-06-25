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

from .glint_algorithms import ThresholdAlgorithm
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
    supports_alignment: bool = False
    # Index of the band to use as the alignment reference. Pick a band with strong texture
    # and high SNR over the expected scene content. Only used when supports_alignment=True.
    alignment_reference_band: int = 0
    # Default alignment strategy when align_bands is True. "rig" uses per-band XMP
    # calibration (only available for MicaSense currently); "phase" uses content-based
    # phase correlation; can be overridden at the call site.
    alignment_strategy: str = "phase"

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Scale the values in the imagery or do other preprocessing logic when overridden."""
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
    ) -> Masker:
        """Create a masker instance for this sensor configuration.

        Parameters
        ----------
        alignment_strategy
            Override the sensor's default alignment strategy. One of "rig", "phase",
            or "none". If None, uses ``self.alignment_strategy``.

        """
        strategy = alignment_strategy if alignment_strategy is not None else self.alignment_strategy
        aligner = None
        if self.supports_alignment and align_bands and strategy != "none":
            from .band_alignment import BandAligner, RigCalibratedAligner  # noqa: PLC0415

            if strategy == "rig":
                aligner = RigCalibratedAligner(reference_band=self.alignment_reference_band)
            elif strategy == "phase":
                aligner = BandAligner(enabled=True, reference_band=self.alignment_reference_band)
            else:
                msg = f"Unknown alignment_strategy: {strategy!r}. Use 'rig', 'phase', or 'none'."
                raise ValueError(msg)

        return Masker(
            algorithm=ThresholdAlgorithm(thresholds, per_band=per_band),
            image_loader=self.loader_class(img_dir, mask_dir),
            image_preprocessor=self.preprocess_image,
            pixel_buffer=pixel_buffer,
            per_band=per_band,
            band_aligner=aligner,
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
)
p4ms_sensor = Sensor(
    name="DJI P4MS",
    bands=[B, G, R, RE, NIR],
    bit_depth=16,
    loader_class=P4MSLoader,
    supports_alignment=True,
)
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
    bands=[B, G, R, NIR, RE],
    bit_depth=16,
    loader_class=MicasenseRedEdgeLoader,
    supports_alignment=True,
    # NIR (index 3) is where glint is most visible — making it the reference means the
    # mask is computed in NIR's coordinate frame, so glint detection is geometrically
    # exact for the band that matters most.
    alignment_reference_band=3,
    alignment_strategy="rig",
)

# RedEdge-MX Dual band order from XMP BandName/CentralWavelength tags. The first 5
# files are the original Camera A (RedEdge-MX) ordering; files _6.._10 are Camera B.
msre_dual_sensor = Sensor(
    name="MicaSense RedEdge-MX Dual",
    bands=[
        Band("Blue 475(32)", 0.875),         # index 0 = file _1
        Band("Green 560(27)", 1.000),         # index 1 = file _2
        Band("Red 668(14)", 1.000),           # index 2 = file _3
        Band("Near-IR 842(57)", 1.000),       # index 3 = file _4
        Band("Red Edge 717(12)", 1.000),      # index 4 = file _5
        Band("Coastal Blue 444(28)", 1.000),  # index 5 = file _6
        Band("Green 531(14)", 1.000),         # index 6 = file _7
        Band("Red 650(16)", 1.000),           # index 7 = file _8
        Band("Red Edge 705(10)", 1.000),      # index 8 = file _9
        Band("Red Edge 740(18)", 1.000),      # index 9 = file _10
    ],
    bit_depth=16,
    loader_class=MicasenseRedEdgeDualLoader,
    supports_alignment=True,
    # NIR (file _4, index 3): glint is most visible there, and it's a Camera A band,
    # so other Camera A bands get sub-pixel residuals from the rig homography.
    alignment_reference_band=3,
    alignment_strategy="rig",
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
