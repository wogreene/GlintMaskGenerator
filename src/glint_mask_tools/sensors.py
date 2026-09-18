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

from .glint_algorithms import SurfaceDiscriminatedThresholdAlgorithm, ThresholdAlgorithm, WhitewashAlgorithm
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

    from .glint_algorithms import GlintAlgorithm


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
    # Band to judge scene-relative contrast in when the contrast rule is enabled.
    # Glint is brightest and water is darkest in NIR, so NIR separates them best;
    # sensors without a NIR band leave this None and fall back to "any band".
    glint_reference_band: int | None = None
    # (numerator, denominator) band indices for the water-column index that tells
    # surface features from anything seen through water. When set, callers can
    # pass ``benthos_index_max`` to create_masker to spare the seafloor from the
    # mask. Pick two bands straddling water's near-infrared absorption rise —
    # Red Edge 717 over NIR 842 on MicaSense — and keep both on the same
    # physical camera so rig alignment registers them.
    benthos_index_bands: tuple[int, int] | None = None
    # Whether the colour + local-contrast whitewash detector applies. It reads
    # bands as R, G, B scaled 0-1, so only plain 3-band visible imagery qualifies.
    supports_whitewash_detector: bool = False
    # Whether captures carry DLS irradiance and are converted to reflectance,
    # so the flight-level irradiance model (and its GUI toggle) applies.
    uses_dls_irradiance: bool = False

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
        benthos_index_max: float | None = None,
        stabilize_irradiance: bool = True,
        mask_saturated: bool = True,
        contrast_multiplier: float | None = None,
        whitewash: dict | None = None,
    ) -> Masker:
        """Create a masker instance for this sensor configuration.

        Parameters
        ----------
        alignment_strategy
            Override the sensor's default alignment strategy. One of "rig", "phase",
            or "none". If None, uses ``self.alignment_strategy``.
        benthos_index_max
            If set (and the sensor declares ``benthos_index_bands``), use the
            surface-discriminated threshold algorithm. Pixels only stay in the
            mask if their water-column index is below this bound, which spares
            anything seen through water (reef, coral, sand) while keeping
            surface glint and whitewash. Around +0.35 for the RedEdge717/NIR842
            pair.
        stabilize_irradiance
            Build a flight-level DLS irradiance model and use it for captures
            whose own sun-sensor geometry is unusable (see the ``irradiance``
            module). Only affects sensors that convert DN to reflectance;
            harmless no-op elsewhere. Disable to get each capture's raw
            per-capture estimate.
        mask_saturated
            Mask pixels that hit the sensor's full well even when their
            converted value falls below the threshold. Clipped pixels convert
            to a floor rather than a measurement — with a long auto-exposure
            that floor can sit below any sensible threshold — so bright glint
            would otherwise drop out of the mask entirely.
        contrast_multiplier
            If set, also mask pixels exceeding this multiple of the capture's
            own background level in ``glint_reference_band``. Unaffected by
            exposure, gain or irradiance, so it keeps working on captures whose
            reflectance scale is compressed by sensor clipping. Around 3 tracks
            visible glint on MicaSense water imagery.
        whitewash
            If set, use the RGB whitewash detector instead of per-band
            thresholds, with these keyword arguments for
            ``WhitewashAlgorithm`` (``{}`` for its defaults). Only for sensors
            with ``supports_whitewash_detector``.

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

        algorithm = self._build_algorithm(
            thresholds,
            per_band=per_band,
            benthos_index_max=benthos_index_max,
            contrast_multiplier=contrast_multiplier,
            whitewash=whitewash,
        )

        from .irradiance import IrradianceCalibrator  # noqa: PLC0415

        return Masker(
            algorithm=algorithm,
            image_loader=self.loader_class(img_dir, mask_dir),
            image_preprocessor=self.preprocess_image,
            pixel_buffer=pixel_buffer,
            per_band=per_band,
            band_aligner=aligner,
            irradiance_calibrator=IrradianceCalibrator(enabled=stabilize_irradiance),
            saturation_dn=self.saturation_dn if mask_saturated else None,
        )

    def _build_algorithm(
        self,
        thresholds: list[float],
        *,
        per_band: bool,
        benthos_index_max: float | None,
        contrast_multiplier: float | None,
        whitewash: dict | None,
    ) -> GlintAlgorithm:
        """Pick the glint algorithm for the requested options."""
        if whitewash is not None:
            if not self.supports_whitewash_detector:
                msg = f"Sensor {self.name!r} doesn't support the whitewash detector."
                raise ValueError(msg)
            return WhitewashAlgorithm(**whitewash, per_band=per_band)

        if benthos_index_max is not None and self.benthos_index_bands is not None:
            numerator_band_idx, denominator_band_idx = self.benthos_index_bands
            return SurfaceDiscriminatedThresholdAlgorithm(
                thresholds,
                numerator_band_idx=numerator_band_idx,
                denominator_band_idx=denominator_band_idx,
                index_max=benthos_index_max,
                per_band=per_band,
                contrast_multiplier=contrast_multiplier,
                reference_band=self.glint_reference_band,
            )
        if benthos_index_max is not None:
            msg = (
                f"Sensor {self.name!r} doesn't declare benthos_index_bands; "
                "benthos_index_max cannot be used."
            )
            raise ValueError(msg)
        return ThresholdAlgorithm(
            thresholds,
            per_band=per_band,
            contrast_multiplier=contrast_multiplier,
            reference_band=self.glint_reference_band,
        )

    @property
    def saturation_dn(self) -> float:
        """Raw DN at or above which a pixel counts as clipped for this sensor."""
        from .radiometric import SATURATION_DN_FRACTION  # noqa: PLC0415

        return SATURATION_DN_FRACTION * ((1 << self.bit_depth) - 1)

    def get_default_thresholds(self) -> list[float]:
        """Get the default threshold values for all bands."""
        return [band.default_threshold for band in self.bands]


rgb_sensor = Sensor(
    name="RGB",
    bands=[R, G, B],
    bit_depth=8,
    loader_class=SingleFileImageLoader,
    supports_whitewash_detector=True,
)
cir_sensor = Sensor(
    name="PhaseOne 4-band CIR",
    bands=[R, G, B, NIR],
    bit_depth=8,
    loader_class=BigTiffLoader,
    glint_reference_band=3,  # Near-IR
    # Red penetrates water much further than NIR, so the same surface-vs-submerged
    # logic applies with Red on top. Untested on this sensor's imagery.
    benthos_index_bands=(0, 3),  # Red, Near-IR
)
p4ms_sensor = Sensor(
    name="DJI P4MS",
    bands=[B, G, R, RE, NIR],
    bit_depth=16,
    loader_class=P4MSLoader,
    supports_alignment=True,
    glint_reference_band=4,  # Near-IR
    # Red Edge over NIR, matching the pair validated on MicaSense. Untested on
    # P4MS reef imagery.
    benthos_index_bands=(3, 4),  # Red Edge, Near-IR
)
# DJI M3M lacks a Blue band, so chromaticity discrimination isn't wired up here.
m3m_sensor = Sensor(
    name="DJI M3M",
    bands=[Band("Green", 0.875), R, RE, NIR],
    bit_depth=16,
    loader_class=DJIM3MLoader,
    supports_alignment=True,
    glint_reference_band=3,  # Near-IR
    # Red Edge over NIR, matching the pair validated on MicaSense. Untested on
    # M3M reef imagery.
    benthos_index_bands=(2, 3),  # Red Edge, Near-IR
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
    uses_dls_irradiance=True,
    # Reflectance scale, not DN — see threshold_max docs on Sensor.
    threshold_max=0.5,
    supports_alignment=True,
    # NIR (index 3) is where glint is most visible — making it the reference means the
    # mask is computed in NIR's coordinate frame, so glint detection is geometrically
    # exact for the band that matters most.
    alignment_reference_band=3,
    alignment_strategy="rig",
    glint_reference_band=3,  # Near-IR 842: glint brightest, water darkest
    # Red Edge 717 (index 4, file _5) over NIR 842 (index 3, file _4) — both on
    # Camera A, so the rig homography registers them sub-pixel. Water absorbs
    # 842 far harder than 717, so anything seen through water reads high.
    benthos_index_bands=(4, 3),
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
    uses_dls_irradiance=True,
    # Reflectance scale, not DN — see threshold_max docs on Sensor.
    threshold_max=0.5,
    supports_alignment=True,
    # NIR (file _4, index 3): glint is most visible there, and it's a Camera A band,
    # so other Camera A bands get sub-pixel residuals from the rig homography.
    alignment_reference_band=3,
    alignment_strategy="rig",
    glint_reference_band=3,  # Near-IR 842: glint brightest, water darkest
    # First 5 files are Camera A (RedEdge-MX), last 5 are Camera B (RedEdge-MX Blue).
    # This lets the aligner fit a single shared rotation for all Camera B bands.
    bands_by_camera=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    # Red Edge 717 (index 4, file _5) over NIR 842 (index 3, file _4), both on
    # Camera A so the rig homography registers them sub-pixel. On Abaco reef
    # imagery this pair kept 93% of labelled glint/whitewash while sparing 99%
    # of labelled reef, coral and sand; a RedEdge/Red pair managed 79%/92% and
    # the original Red/Blue pair 63%/87%.
    benthos_index_bands=(4, 3),
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
