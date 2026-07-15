"""Band alignment module for multi-band imagery.

Two aligner strategies live here:

* :class:`BandAligner` — content-driven phase correlation. Works on any multi-band
  sensor without sensor-specific metadata. Estimates a single translation offset
  per band from a sample of calibration images.
* :class:`RigCalibratedAligner` — calibrated geometric warp built from per-band
  XMP metadata (focal length, principal point, rig rotation relatives, optional
  lens distortion). Accurate to sub-pixel within a single camera; for multi-camera
  rigs like MicaSense RedEdge-MX Dual, the inter-camera baseline leaves a small
  residual that can be cleaned up by an additional phase-correlation pass.

Created by: Taylor Denouden
Organization: Hakai Institute
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import tifffile
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

# Fraction of the sequence to keep in the middle when picking calibration samples.
# Trims takeoff/ascent at the start and landing/descent at the end where altitude
# and ground content aren't representative of the survey.
_MIDDLE_FRACTION = 0.6

# phaseCorrelate response below this is considered unreliable and the measurement
# is dropped before taking the median across samples. Empirically, clean correlations
# return >0.1 and noise floors sit around 0.01-0.03.
_DEFAULT_MIN_RESPONSE = 0.05


def _pick_middle_sample_indices(total: int, n: int) -> list[int]:
    """Pick `n` indices evenly spaced across the middle ``_MIDDLE_FRACTION`` of a sequence.

    Trims the first/last fraction of the sequence so takeoff/ascent and landing/descent
    frames don't poison sample-based calibration.
    """
    n = min(n, total)
    if total <= n:
        return list(range(total))
    margin = (1.0 - _MIDDLE_FRACTION) / 2.0
    lo = int(round(total * margin))
    hi = int(round(total * (1.0 - margin))) - 1
    if hi <= lo:
        return list(range(total))[:n]
    return [int(round(lo + i * (hi - lo) / max(n - 1, 1))) for i in range(n)]


@dataclass
class BandOffsets:
    """Calibrated offsets for each band relative to the reference band."""

    x_offsets: tuple[float, ...]
    y_offsets: tuple[float, ...]

    @property
    def num_bands(self) -> int:
        """Return the number of bands."""
        return len(self.x_offsets)

    def has_offset(self) -> bool:
        """Return True if any band has a non-trivial offset (>= 0.5 px in either axis)."""
        return any(abs(x) >= 0.5 or abs(y) >= 0.5 for x, y in zip(self.x_offsets, self.y_offsets))  # noqa: PLR2004


class BandAligner:
    """Handles band alignment calibration and application using phase correlation."""

    def __init__(
        self,
        calibration_samples: int = 5,
        *,
        enabled: bool = True,
        reference_band: int = 0,
        min_response: float = _DEFAULT_MIN_RESPONSE,
    ) -> None:
        """Create a new BandAligner.

        Parameters
        ----------
        calibration_samples
            Number of images to sample for calibration (default 5).
        enabled
            Whether alignment is enabled (default True).
        reference_band
            Index of the band to use as the alignment reference. All other bands
            are shifted to match this one. Default 0.
        min_response
            Minimum phaseCorrelate response value for a measurement to be trusted.
            Measurements below this are dropped before taking the median.

        """
        self.calibration_samples = calibration_samples
        self.enabled = enabled
        self.reference_band = reference_band
        self.min_response = min_response
        self._offsets: BandOffsets | None = None
        self._calibrated = False

    @property
    def is_calibrated(self) -> bool:
        """Return True if calibration has been performed."""
        return self._calibrated

    @property
    def offsets(self) -> BandOffsets | None:
        """Return the calibrated offsets, or None if not calibrated."""
        return self._offsets

    def _pick_sample_indices(self, total: int) -> list[int]:
        """Pick calibration sample indices from the middle of the sequence."""
        return _pick_middle_sample_indices(total, self.calibration_samples)

    def calibrate(
        self,
        image_paths: Iterable[list[str] | str],
        load_fn: Callable[[list[str] | str], np.ndarray],
    ) -> BandOffsets | None:
        """Calibrate alignment offsets from a sample of images.

        Parameters
        ----------
        image_paths
            Iterable of image paths (or path groups for multi-file loaders).
        load_fn
            Function to load image data given paths.

        Returns
        -------
        BandOffsets | None
            Calibrated offsets for each band, or None if calibration failed.

        """
        if not self.enabled:
            self._calibrated = True
            return None

        paths_list = list(image_paths)
        if not paths_list:
            logger.warning("No images found for band alignment calibration")
            self._calibrated = True
            return None

        sample_indices = self._pick_sample_indices(len(paths_list))
        sample_paths = [paths_list[i] for i in sample_indices]
        logger.info(
            f"Calibrating band alignment from {len(sample_paths)} sample images "
            f"(indices {sample_indices} of {len(paths_list)}, reference band={self.reference_band})"
        )

        # Per-band lists of (x_offset, y_offset) measurements that passed the response check.
        per_band_measurements: list[list[tuple[float, float]]] = []

        try:
            for paths in sample_paths:
                img = load_fn(paths)

                if img.ndim != 3:  # noqa: PLR2004
                    logger.warning("Image is not multi-band, skipping alignment calibration")
                    self.enabled = False
                    self._calibrated = True
                    return None

                num_bands = img.shape[2]
                if not per_band_measurements:
                    per_band_measurements = [[] for _ in range(num_bands)]

                if not 0 <= self.reference_band < num_bands:
                    msg = f"reference_band={self.reference_band} out of range for {num_bands} bands"
                    raise ValueError(msg)

                ref_band = img[:, :, self.reference_band]

                for band_idx in range(num_bands):
                    if band_idx == self.reference_band:
                        per_band_measurements[band_idx].append((0.0, 0.0))
                        continue
                    target_band = img[:, :, band_idx]
                    x_off, y_off, response = self._estimate_offset(ref_band, target_band)
                    if response < self.min_response:
                        logger.debug(
                            f"Dropping band {band_idx} measurement: response {response:.4f} < {self.min_response}"
                        )
                        continue
                    per_band_measurements[band_idx].append((x_off, y_off))

        except cv2.error as e:
            logger.warning(f"Band alignment calibration failed: {e}. Alignment disabled.")
            self.enabled = False
            self._calibrated = True
            return None

        if per_band_measurements:
            num_bands = len(per_band_measurements)
            median_x: list[float] = []
            median_y: list[float] = []
            for band_idx, measurements in enumerate(per_band_measurements):
                if not measurements:
                    logger.warning(
                        f"No reliable phase correlation measurements for band {band_idx}; assuming zero offset"
                    )
                    median_x.append(0.0)
                    median_y.append(0.0)
                    continue
                xs = [m[0] for m in measurements]
                ys = [m[1] for m in measurements]
                median_x.append(float(np.median(xs)))
                median_y.append(float(np.median(ys)))

            self._offsets = BandOffsets(x_offsets=tuple(median_x), y_offsets=tuple(median_y))

            if self._offsets.has_offset():
                x_fmt = "(" + ", ".join(f"{v:.2f}" for v in median_x) + ")"
                y_fmt = "(" + ", ".join(f"{v:.2f}" for v in median_y) + ")"
                logger.info(f"Band alignment offsets: x={x_fmt}, y={y_fmt}")
            else:
                logger.info("Band alignment calibration found no significant offsets")

        self._calibrated = True
        return self._offsets

    def align(self, img: np.ndarray) -> np.ndarray:
        """Apply calibrated alignment to an image.

        Parameters
        ----------
        img
            Image array with shape (H, W, num_bands).

        Returns
        -------
        np.ndarray
            Aligned image with same shape.

        Raises
        ------
        ValueError
            If image band count doesn't match calibration.

        """
        if not self.enabled:
            return img

        if self._offsets is None:
            return img

        if not self._offsets.has_offset():
            return img

        if img.shape[2] != self._offsets.num_bands:
            msg = f"Image has {img.shape[2]} bands but alignment calibrated for {self._offsets.num_bands} bands"
            raise ValueError(msg)

        aligned_bands = []
        for band_idx in range(img.shape[2]):
            band = img[:, :, band_idx]
            x_off = self._offsets.x_offsets[band_idx]
            y_off = self._offsets.y_offsets[band_idx]
            aligned_band = self._apply_offset(band, x_off, y_off)
            aligned_bands.append(aligned_band)

        return np.stack(aligned_bands, axis=2)

    def unalign_mask(self, mask: np.ndarray) -> np.ndarray:
        """Shift a mask back to original (unaligned) coordinate space for each band.

        Takes a 2D union mask (or 3D per-band masks) and creates shifted versions
        for each band so the mask aligns with the original unaligned band images.

        Parameters
        ----------
        mask
            Mask array with shape (H, W) for union mask or (H, W, num_bands) for per-band.

        Returns
        -------
        np.ndarray
            3D mask array (H, W, num_bands) with each band shifted to its original coordinates.

        """
        if not self.enabled:
            return mask

        if self._offsets is None:
            return mask

        if not self._offsets.has_offset():
            return mask

        # Handle 2D union mask - replicate and shift for each band
        if mask.ndim == 2:  # noqa: PLR2004
            logger.info(
                f"Shifting union mask to per-band coordinates: x={self._offsets.x_offsets}, y={self._offsets.y_offsets}"
            )
            unaligned_masks = []
            for band_idx in range(self._offsets.num_bands):
                x_off = -self._offsets.x_offsets[band_idx]
                y_off = -self._offsets.y_offsets[band_idx]
                unaligned_mask = self._apply_offset(mask.astype(np.float32), x_off, y_off)
                unaligned_masks.append(unaligned_mask)
            return np.stack(unaligned_masks, axis=2)

        # Handle 3D per-band masks
        if mask.shape[2] != self._offsets.num_bands:
            msg = f"Mask has {mask.shape[2]} bands but alignment calibrated for {self._offsets.num_bands} bands"
            raise ValueError(msg)

        logger.info(
            f"Shifting per-band masks to original coordinates: x={self._offsets.x_offsets}, y={self._offsets.y_offsets}"
        )

        unaligned_masks = []
        for band_idx in range(mask.shape[2]):
            mask_band = mask[:, :, band_idx]
            x_off = -self._offsets.x_offsets[band_idx]
            y_off = -self._offsets.y_offsets[band_idx]
            unaligned_mask = self._apply_offset(mask_band.astype(np.float32), x_off, y_off)
            unaligned_masks.append(unaligned_mask)

        return np.stack(unaligned_masks, axis=2)

    @staticmethod
    def _estimate_offset(
        ref_band: np.ndarray,
        target_band: np.ndarray,
    ) -> tuple[float, float, float]:
        """Estimate sub-pixel offset of target band relative to reference using phase correlation.

        Returns
        -------
        tuple[float, float, float]
            (x_offset, y_offset, response). The offsets are the sub-pixel correction
            vector that should be applied to target_band to align it with ref_band.
            response is phaseCorrelate's peak response value (higher = more reliable).

        """
        shift, response = cv2.phaseCorrelate(
            ref_band.astype(np.float64),
            target_band.astype(np.float64),
        )
        # Negate to get correction vector (shift needed to align target to ref).
        # Keep sub-pixel precision — warpAffine handles fractional translations.
        return -float(shift[0]), -float(shift[1]), float(response)

    @staticmethod
    def _apply_offset(
        band: np.ndarray,
        x_offset: float,
        y_offset: float,
    ) -> np.ndarray:
        """Apply translation offset to a single band (sub-pixel via bilinear interpolation)."""
        if x_offset == 0 and y_offset == 0:
            return band

        m = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        return cv2.warpAffine(
            band.astype(np.float32),
            m,
            (band.shape[1], band.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )


# --------------------------------------------------------------------------------------
# Rig-calibrated alignment
# --------------------------------------------------------------------------------------

# MicaSense RedEdge sensors (RedEdge-M, RedEdge-MX, RedEdge-MX Dual) all share the
# same 1280x960 imager with a 4.8x3.6 mm active area, i.e. 3.75 µm pixel pitch.
# Used to convert mm-valued XMP tags into pixel coordinates.
_MICASENSE_PIXEL_PITCH_MM = 0.00375

# XMP namespaces. The Pix4D Camera schema is shared by MicaSense and Parrot Sequoia.
_XMP_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "Camera": "http://pix4d.com/camera/1.0",
}


@dataclass(frozen=True)
class BandCalibration:
    """Per-band geometric calibration parsed from XMP metadata."""

    band_idx: int
    band_name: str
    central_wavelength_nm: float
    rig_camera_index: int
    focal_length_mm: float
    principal_point_mm: tuple[float, float]
    rig_relatives_deg: tuple[float, float, float]
    distortion: tuple[float, float, float, float, float] | None
    image_size: tuple[int, int]  # (W, H)
    pixel_pitch_mm: float = _MICASENSE_PIXEL_PITCH_MM

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix in pixel coordinates."""
        f_px = self.focal_length_mm / self.pixel_pitch_mm
        cx_px = self.principal_point_mm[0] / self.pixel_pitch_mm
        cy_px = self.principal_point_mm[1] / self.pixel_pitch_mm
        return np.array(
            [[f_px, 0.0, cx_px], [0.0, f_px, cy_px], [0.0, 0.0, 1.0]], dtype=np.float64
        )


def parse_micasense_xmp(path: str | Path, band_idx: int) -> BandCalibration:
    """Read MicaSense (Pix4D-schema) calibration metadata from a single band TIFF.

    Raises
    ------
    ValueError
        If required Camera tags are missing or malformed.

    """
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        if "XMP" not in page.tags:
            msg = f"No XMP tag in {path}"
            raise ValueError(msg)
        xmp = page.tags["XMP"].value
        if isinstance(xmp, bytes):
            xmp_str = xmp.decode("utf-8", errors="ignore")
        else:
            xmp_str = str(xmp)
        h, w = page.shape[:2]

    try:
        root = ET.fromstring(xmp_str)
    except ET.ParseError as e:
        msg = f"Malformed XMP in {path}: {e}"
        raise ValueError(msg) from e

    # The Pix4D camera tags live in their own rdf:Description block. Identify it by
    # the presence of Camera:BandName.
    camera_desc = None
    for desc in root.findall(".//rdf:Description", _XMP_NS):
        if desc.find("Camera:BandName", _XMP_NS) is not None:
            camera_desc = desc
            break
    if camera_desc is None:
        msg = f"No Pix4D Camera description block in {path}"
        raise ValueError(msg)

    def _text(name: str, *, required: bool = True) -> str | None:
        el = camera_desc.find(f"Camera:{name}", _XMP_NS)
        if el is not None and el.text is not None:
            return el.text.strip()
        if required:
            msg = f"missing tag Camera:{name} in {path}"
            raise ValueError(msg)
        return None

    def _seq_floats(name: str) -> list[float] | None:
        el = camera_desc.find(f"Camera:{name}", _XMP_NS)
        if el is None:
            return None
        seq = el.find("rdf:Seq", _XMP_NS)
        if seq is None:
            return None
        return [float(li.text) for li in seq.findall("rdf:li", _XMP_NS) if li.text is not None]

    band_name = _text("BandName")
    central_wavelength = float(_text("CentralWavelength"))
    rig_camera_index = int(_text("RigCameraIndex"))
    focal_length_mm = float(_text("PerspectiveFocalLength"))

    pp_parts = [float(x) for x in _text("PrincipalPoint").split(",")]
    if len(pp_parts) != 2:  # noqa: PLR2004
        msg = f"PrincipalPoint malformed in {path}"
        raise ValueError(msg)
    principal_point_mm = (pp_parts[0], pp_parts[1])

    rr_parts = [float(x) for x in _text("RigRelatives").split(",")]
    if len(rr_parts) != 3:  # noqa: PLR2004
        msg = f"RigRelatives malformed in {path}"
        raise ValueError(msg)
    rig_relatives_deg = (rr_parts[0], rr_parts[1], rr_parts[2])

    dist_seq = _seq_floats("PerspectiveDistortion")
    distortion = tuple(dist_seq) if dist_seq is not None and len(dist_seq) == 5 else None  # noqa: PLR2004

    return BandCalibration(
        band_idx=band_idx,
        band_name=band_name,
        central_wavelength_nm=central_wavelength,
        rig_camera_index=rig_camera_index,
        focal_length_mm=focal_length_mm,
        principal_point_mm=principal_point_mm,
        rig_relatives_deg=rig_relatives_deg,
        distortion=distortion,
        image_size=(w, h),
    )


def _euler_to_rotation_xyz(angles_deg: tuple[float, float, float]) -> np.ndarray:
    """3x3 rotation matrix from extrinsic XYZ Euler angles (degrees).

    Empirically the convention matching MicaSense's RigRelatives on RedEdge-MX Dual:
    each band's tag is the rotation that takes vectors *from the rig body frame into
    that band's optical frame* (so going from band B to ref needs ``R_ref^T @ R_B``).
    """
    a, b, c = np.deg2rad(angles_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rx @ ry @ rz


class RigCalibratedAligner:
    """Aligner using per-band XMP calibration (rig rotation + intrinsics).

    Treats the scene as being at infinity, so only the rotation between bands and
    differences in intrinsics (focal length, principal point) contribute to the
    warp. This assumption is excellent for aerial imagery at typical survey
    altitudes — the inter-camera baseline is ~5 cm against ~60 m altitude, giving
    sub-pixel parallax.

    The calibration is parsed once from the first capture; XMP tags are written
    identically to every TIFF in a flight, so a single read is sufficient.
    """

    def __init__(
        self,
        reference_band: int = 0,
        *,
        enabled: bool = True,
        parse_xmp: Callable[[str | Path, int], BandCalibration] = parse_micasense_xmp,
        apply_distortion_correction: bool = False,
        refinement_method: str = "translation",
        refinement_samples: int = 8,
        refinement_min_response: float = 0.05,
        refinement_min_cc: float = 0.4,
        ecc_max_iterations: int = 50,
        skip_ecc_below_px: float = 1.0,
        foreign_camera_bands: list[int] | None = None,
    ) -> None:
        """Create a new RigCalibratedAligner.

        Parameters
        ----------
        reference_band
            Index of the band into whose frame all others are warped.
        enabled
            If False, behaves as a no-op (returns inputs unchanged).
        parse_xmp
            Callable that reads per-band metadata from a TIFF path. Defaults to
            the MicaSense Pix4D-schema parser. Pluggable so future sensors with
            different schemas (Altum, P4MS) can supply their own parser.
        apply_distortion_correction
            If True, undistort each band with its own distortion coefficients before
            applying the rotation homography. Slightly more accurate near image
            edges but slower; off by default since the residual is small for
            MicaSense's near-rectilinear lenses.
        refinement_method
            Per-flight residual refinement, applied once during calibration and
            baked into each band's homography. One of:

            - ``"none"`` — no refinement; rig homography is applied as-is.
            - ``"translation"`` (default) — sample N captures, phase-correlate
              each band against the reference, take the median translation, fold
              it in. Catches the constant inter-camera offset that the rig
              metadata doesn't fully model.
            - ``"affine"`` — "translation" plus a **shared** rotation applied
              to every band on a foreign physical camera (as declared by
              ``foreign_camera_bands``). All bands on a single PCB share the
              same rigid rotation offset relative to the reference camera, so a
              per-camera rotation median is more robust than fitting per-band.
              Per-band translation still handles the DC offset. Requires
              ``foreign_camera_bands`` to be non-empty; otherwise falls back to
              translation-only behavior.
        refinement_samples
            Number of mid-flight captures to sample for refinement fitting.
        refinement_min_response
            phaseCorrelate response below which a sample's translation
            measurement is dropped before taking the per-band median.
        refinement_min_cc
            ECC correlation-coefficient below which a sample's ECC fit is
            rejected before taking the per-band median.
        ecc_max_iterations
            Maximum ECC optimizer iterations per fit during calibration.
        skip_ecc_below_px
            If the phase-correlation translation residual is already below this
            many pixels for a band, skip the ECC step for that sample and just
            record the translation. On MicaSense Dual this catches the four
            Camera A bands, where the rig homography is already sub-pixel
            accurate. No effect if ``refinement_method`` is "none" or
            "translation".
        foreign_camera_bands
            Band indices on a physical camera other than the reference band's.
            Only these bands get the shared rotation from ``refinement_method="affine"``.
            E.g. for MicaSense Dual with reference on Camera A: [5, 6, 7, 8, 9].
            If None or empty, "affine" degrades to translation-only behavior.

        """
        valid_methods = {"none", "translation", "affine"}
        if refinement_method not in valid_methods:
            msg = f"refinement_method must be one of {valid_methods}, got {refinement_method!r}"
            raise ValueError(msg)

        self.reference_band = reference_band
        self.enabled = enabled
        self._parse_xmp = parse_xmp
        self.apply_distortion_correction = apply_distortion_correction
        self.refinement_method = refinement_method
        self.refinement_samples = refinement_samples
        self.refinement_min_response = refinement_min_response
        self.refinement_min_cc = refinement_min_cc
        self.ecc_max_iterations = ecc_max_iterations
        self.skip_ecc_below_px = skip_ecc_below_px
        self.foreign_camera_bands = set(foreign_camera_bands) if foreign_camera_bands else set()

        self._calibrated = False
        self._calibrations: list[BandCalibration] | None = None
        self._homographies: list[np.ndarray] | None = None
        self._homographies_inv: list[np.ndarray] | None = None
        self._size: tuple[int, int] | None = None  # (W, H)

    @property
    def is_calibrated(self) -> bool:
        """Return True if calibration has been performed."""
        return self._calibrated

    @property
    def num_bands(self) -> int | None:
        """Return the number of bands the aligner is calibrated for."""
        return len(self._calibrations) if self._calibrations else None

    def calibrate(
        self,
        image_paths: Iterable[list[str] | str],
        load_fn: Callable[[list[str] | str], np.ndarray] | None = None,
    ) -> None:
        """Parse XMP from the first capture and compute per-band homographies.

        Then (if ``refinement_method`` is not "none") sample a handful of
        mid-flight captures, fit the residual per band, take the median, and
        fold it into each band's homography. All refinement happens here;
        apply-time is just rig warps.
        """
        if not self.enabled:
            self._calibrated = True
            return

        paths_list = list(image_paths)
        if not paths_list:
            logger.warning("No images found for rig calibration")
            self._calibrated = True
            return

        first = paths_list[0]
        if isinstance(first, str):
            msg = "RigCalibratedAligner requires per-band file paths (multi-file loader)"
            logger.warning(msg)
            self.enabled = False
            self._calibrated = True
            return

        try:
            self._calibrations = [self._parse_xmp(p, i) for i, p in enumerate(first)]
        except (ValueError, OSError) as e:
            logger.warning(f"Rig calibration failed: {e}. Falling back to no-op alignment.")
            self.enabled = False
            self._calibrated = True
            return

        if not 0 <= self.reference_band < len(self._calibrations):
            msg = (
                f"reference_band={self.reference_band} out of range for "
                f"{len(self._calibrations)} bands"
            )
            raise ValueError(msg)

        # All bands in a MicaSense capture share the same image size.
        self._size = self._calibrations[self.reference_band].image_size
        self._build_homographies()

        if self.refinement_method != "none" and load_fn is not None and len(paths_list) > 1:
            self._refine_per_flight(paths_list, load_fn)

        self._calibrated = True

        logger.info(
            f"RigCalibratedAligner: {len(self._calibrations)} bands, "
            f"reference={self.reference_band} ({self._calibrations[self.reference_band].band_name}), "
            f"size={self._size}, refinement={self.refinement_method}"
        )

    def _refine_per_flight(
        self,
        paths_list: list[list[str] | str],
        load_fn: Callable[[list[str] | str], np.ndarray],
    ) -> None:
        """Sample N captures, fit the per-band residual on each, fold the median in.

        For ``refinement_method="translation"`` we fit phase-correlation shifts;
        for "affine" we fit ECC EUCLIDEAN (rotation + translation). The result is
        a single fixed correction per band, applied to every capture.
        """
        assert self._homographies is not None
        assert self._calibrations is not None

        sample_indices = _pick_middle_sample_indices(len(paths_list), self.refinement_samples)
        # Store per-sample (theta, tx, ty). For translation-only, theta is always 0.
        per_band_fits: list[list[tuple[float, float, float]]] = [[] for _ in self._calibrations]

        try:
            for idx in sample_indices:
                img = load_fn(paths_list[idx])
                aligned = self._apply_warps(img, self._homographies)
                ref_band = aligned[:, :, self.reference_band]
                for i in range(len(self._calibrations)):
                    if i == self.reference_band:
                        continue
                    fit = self._fit_sample(ref_band, aligned[:, :, i])
                    if fit is not None:
                        per_band_fits[i].append(fit)
        except (cv2.error, OSError) as e:
            logger.warning(f"Per-flight refinement failed: {e}. Using rig only.")
            return

        # Shared rotation across all foreign-camera bands. Pooling ECC theta fits
        # across every foreign band and every sample gives a robust estimate of
        # the fixed rotational offset between the two physical camera PCBs. Only
        # samples where ECC actually ran (non-zero theta) contribute — the ones
        # that hit the skip_ecc_below_px fast path carry no rotation signal.
        #
        # Sign note: ECC's returned warp W maps pre_warped → ref. When we
        # decompose W into rotation-about-origin theta + translation, then
        # rebuild it as a rotation-about-image-center transform applied to the
        # raw target, the sign of the rotation must be *negated*. Composing with
        # ECC's own theta rotates in the same direction as the existing rig
        # misalignment (making it worse); we need the opposite direction to
        # undo it. Verified empirically via tile-based residual measurement.
        shared_theta = 0.0
        if self.refinement_method == "affine" and self.foreign_camera_bands:
            pool = [
                f[0]
                for i, fits in enumerate(per_band_fits)
                if i in self.foreign_camera_bands
                for f in fits
                if abs(f[0]) > 1e-6
            ]
            if pool:
                shared_theta = -float(np.median(pool))
                logger.info(
                    f"Shared foreign-camera rotation: theta={np.degrees(shared_theta):+.3f}deg "
                    f"(from {len(pool)} ECC fits across bands {sorted(self.foreign_camera_bands)})"
                )

        cx = self._size[0] * 0.5
        cy = self._size[1] * 0.5
        refined: list[np.ndarray] = []
        refined_inv: list[np.ndarray] = []
        report: list[str] = []
        for i, h in enumerate(self._homographies):
            fits = per_band_fits[i]
            if i == self.reference_band or not fits:
                refined.append(h)
                refined_inv.append(np.linalg.inv(h))
                report.append("(ref)" if i == self.reference_band else "(no fit)")
                continue
            median_tx = float(np.median([f[1] for f in fits]))
            median_ty = float(np.median([f[2] for f in fits]))
            # Only foreign-camera bands get the shared rotation; ref-camera bands
            # get translation only (rig calibration already aligns their rotation).
            theta = shared_theta if i in self.foreign_camera_bands else 0.0
            c, s = np.cos(theta), np.sin(theta)
            correction = np.array(
                [
                    [c, -s, cx * (1 - c) + cy * s + median_tx],
                    [s, c, -cx * s + cy * (1 - c) + median_ty],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            h_final = correction @ h
            refined.append(h_final)
            refined_inv.append(np.linalg.inv(h_final))
            report.append(f"({np.degrees(theta):+.2f}deg, {median_tx:+.1f}, {median_ty:+.1f})")
        self._homographies = refined
        self._homographies_inv = refined_inv

        logger.info(f"Per-flight refinement (theta,tx,ty) per band: {', '.join(report)}")

    def _fit_sample(
        self,
        ref_band: np.ndarray,
        target_band: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """Fit a single sample capture. Returns (theta_rad, tx_px, ty_px) or None.

        Phase-correlation gives a robust translation estimate. If ``refinement_method``
        is "affine", we also run ECC to recover the rotation but *only* use ECC's
        rotation component — its translation output covaries with any per-capture
        alignment noise and would inflate the variance of the per-band median.
        Keeping translation and rotation decoupled lets each dimension's median
        converge independently.
        """
        shift, response = cv2.phaseCorrelate(
            ref_band.astype(np.float64), target_band.astype(np.float64)
        )
        if response < self.refinement_min_response:
            return None
        tx, ty = -float(shift[0]), -float(shift[1])

        if self.refinement_method == "translation":
            return 0.0, tx, ty

        # affine: skip ECC when phase-corr residual is already tiny (Camera A path).
        if tx * tx + ty * ty < self.skip_ecc_below_px * self.skip_ecc_below_px:
            return 0.0, tx, ty

        ref32 = ref_band.astype(np.float32)
        t_2x3 = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
        pre_warped = cv2.warpAffine(
            target_band.astype(np.float32),
            t_2x3,
            (ref32.shape[1], ref32.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.ecc_max_iterations, 1e-5)
        try:
            cc, warp = cv2.findTransformECC(
                ref32, pre_warped, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5,
            )
        except cv2.error:
            return 0.0, tx, ty
        if cc < self.refinement_min_cc:
            return 0.0, tx, ty

        # Extract only the rotation from ECC; keep translation from phase corr.
        c, s = float(warp[0, 0]), float(warp[1, 0])
        theta = float(np.arctan2(s, c))
        return theta, tx, ty

    def _build_homographies(self) -> None:
        assert self._calibrations is not None
        ref = self._calibrations[self.reference_band]
        k_ref = ref.K
        r_ref = _euler_to_rotation_xyz(ref.rig_relatives_deg)

        self._homographies = []
        self._homographies_inv = []
        for band in self._calibrations:
            if band.band_idx == self.reference_band:
                self._homographies.append(np.eye(3))
                self._homographies_inv.append(np.eye(3))
                continue
            r_b = _euler_to_rotation_xyz(band.rig_relatives_deg)
            r_rel = r_ref.T @ r_b  # band frame → ref frame
            h = k_ref @ r_rel @ np.linalg.inv(band.K)
            self._homographies.append(h)
            self._homographies_inv.append(np.linalg.inv(h))

    def align(self, img: np.ndarray) -> np.ndarray:
        """Warp each band's image into the reference band's coordinate frame."""
        if not self.enabled or not self._calibrated or self._homographies is None:
            return img
        return self._apply_warps(img, self._homographies)

    def _apply_warps(self, img: np.ndarray, homographies: list[np.ndarray]) -> np.ndarray:
        """Apply per-band homography warps to a multi-band image.

        Separate from ``align`` so refinement can call it before _calibrated is True.
        """
        if img.shape[2] != len(homographies):
            msg = (
                f"Image has {img.shape[2]} bands but rig calibration is for "
                f"{len(homographies)} bands"
            )
            raise ValueError(msg)

        w, h = self._size
        out = np.empty_like(img, dtype=np.float32)
        for i, mat in enumerate(homographies):
            if i == self.reference_band:
                out[:, :, i] = img[:, :, i].astype(np.float32)
                continue
            out[:, :, i] = cv2.warpPerspective(
                img[:, :, i].astype(np.float32),
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return out

    def unalign_mask(self, mask: np.ndarray) -> np.ndarray:
        """Warp a union mask back into each band's original coordinate frame.

        Returns a 3D mask (H, W, num_bands) regardless of input dimensionality,
        because each band's geometric warp is distinct.

        Accepts either a 2D union mask (replicated and warped per band) or a 3D
        per-band mask (each slice warped with its own inverse homography).
        """
        if not self.enabled or not self._calibrated or self._homographies_inv is None:
            return mask

        homographies_inv = self._homographies_inv
        n_bands = len(homographies_inv)
        w, h = self._size

        if mask.ndim == 3:  # noqa: PLR2004
            if mask.shape[2] != n_bands:
                msg = (
                    f"Mask has {mask.shape[2]} bands but rig calibration is for {n_bands} bands"
                )
                raise ValueError(msg)
            per_band_sources = [mask[:, :, i].astype(np.float32) for i in range(n_bands)]
        else:
            shared = mask.astype(np.float32)
            per_band_sources = [shared for _ in range(n_bands)]

        out = np.empty((h, w, n_bands), dtype=np.float32)
        for i, mat in enumerate(homographies_inv):
            if i == self.reference_band:
                out[:, :, i] = per_band_sources[i]
                continue
            out[:, :, i] = cv2.warpPerspective(
                per_band_sources[i],
                mat,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return out
