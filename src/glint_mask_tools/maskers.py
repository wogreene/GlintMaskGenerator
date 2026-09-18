"""Module containing the main Masker classes for different comninations of glint masking algorithms and _known_sensors.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
"""

from __future__ import annotations

import concurrent.futures
import os
from typing import TYPE_CHECKING, Callable

import numpy as np
from loguru import logger
from PIL import Image
from scipy.ndimage import convolve

from .utils import make_circular_kernel

if TYPE_CHECKING:
    from .band_alignment import BandAligner
    from .glint_algorithms import GlintAlgorithm
    from .image_loaders import ImageLoader
    from .irradiance import IrradianceCalibrator


# Peak memory per image sample (pixel x band) while one capture is being masked,
# measured on 45 MP RGB drone photos (~1.9 GB each) and rounded up for headroom.
_PEAK_BYTES_PER_SAMPLE = 16

# Share of physical memory all workers together may use, leaving the rest for
# the OS and whatever else is running.
_MEMORY_BUDGET_FRACTION = 0.6


def physical_memory_bytes() -> int | None:
    """Total physical memory, or None where it can't be read (e.g. Windows)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


class Masker:
    """The main class for masking glint in imagery. It is composed of an image loader and glint masking algorithm."""

    def __init__(  # noqa: PLR0913
        self,
        algorithm: GlintAlgorithm,
        image_loader: ImageLoader,
        image_preprocessor: Callable[[np.ndarray], np.ndarray],
        pixel_buffer: int = 0,
        *,
        per_band: bool = False,
        band_aligner: BandAligner | None = None,
        irradiance_calibrator: IrradianceCalibrator | None = None,
        saturation_dn: float | None = None,
    ) -> None:
        """Create the Masker object.

        ``saturation_dn`` is the raw DN at which the sensor is considered
        clipped; pixels at or above it are masked regardless of the threshold
        (their converted value is a floor, not a measurement). Pass None to
        disable that behaviour and mask purely on the threshold.
        """
        self.algorithm = algorithm
        self.image_loader = image_loader
        self.image_preprocessor = image_preprocessor
        self.pixel_buffer = pixel_buffer
        self.per_band = per_band
        self.band_aligner = band_aligner
        self.irradiance_calibrator = irradiance_calibrator
        self.saturation_dn = saturation_dn
        self.buffer_kernel = make_circular_kernel(self.pixel_buffer)

    # noinspection PyMethodMayBeStatic
    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Postprocess the generated boolean numpy mask. Can be overridden to customize behavior."""
        if self.pixel_buffer <= 0:
            return mask
        if mask.ndim == 2:  # noqa: PLR2004
            # Single combined mask
            return convolve(mask, self.buffer_kernel, mode="constant", cval=0) > 0
        # Per-band masks: apply buffer to each channel
        result = np.empty_like(mask, dtype=bool)
        for i in range(mask.shape[2]):
            result[:, :, i] = convolve(mask[:, :, i], self.buffer_kernel, mode="constant", cval=0) > 0
        return result

    @staticmethod
    def to_metashape_mask(mask: np.ndarray) -> np.ndarray:
        """Scale the mask values to work with Agisoft Metashape expectations."""
        return np.logical_not(mask).astype(np.uint8) * 255

    def __len__(self) -> int:
        """Get and return the number of files to process.

        Returns
        -------
        int
            The number of files that need to be processed.

        """
        return len(self.image_loader)

    def _calibrate_alignment(self) -> None:
        """Calibrate band alignment if aligner is configured."""
        if self.band_aligner is None:
            return
        if self.band_aligner.is_calibrated:
            return

        self.band_aligner.calibrate(
            image_paths=self.image_loader.paths,
            load_fn=self.image_loader.load_image,
        )

    def _calibrate_irradiance(self) -> None:
        """Build the flight-level DLS irradiance model if a calibrator is configured."""
        if self.irradiance_calibrator is None:
            return
        if self.irradiance_calibrator.calibration_attempted:
            return

        self.irradiance_calibrator.calibrate(
            capture_paths=self.image_loader.paths,
            read_metadata_fn=self.image_loader.read_radiometric_metadata,
        )
        # Same sweep already holds the metadata needed to tell whether the
        # thresholds are even reachable, which is worth saying out loud before
        # a long run silently produces empty masks.
        thresholds = getattr(self.algorithm, "thresholds", None)
        if thresholds is not None:
            self.irradiance_calibrator.warn_on_unreachable_thresholds(thresholds)

    def __call__(
        self,
        max_workers: int,
        callback: Callable[[list[str]], None] = lambda _: None,
        err_callback: Callable[[list[str], Exception], None] = lambda _s, _e: None,
    ) -> None:
        """Run the masker processing.

        Parameters
        ----------
        max_workers
            The number of threads to use during processing. Useful for limiting memory
            consumption.
        callback
            Callback that receives the img_path as an arg after it is
            successfully processed.
        err_callback
            Callback that receives the img_path and an Exception as args after
            a processing failure.

        """
        self._calibrate_alignment()
        self._calibrate_irradiance()
        max_workers = self.memory_limited_workers(max_workers)

        if max_workers == 0:
            return self.process_unthreaded(callback, err_callback)
        return self.process(max_workers, callback, err_callback)

    def _capture_samples(self) -> int | None:
        """Pixels x bands in one capture, read from file headers only."""
        first = next(iter(self.image_loader.paths), None)
        if first is None:
            return None
        total = 0
        try:
            for path in [first] if isinstance(first, str) else first:
                with Image.open(path) as im:
                    width, height = im.size
                    total += width * height * len(im.getbands())
        except Exception:
            return None
        return total

    def memory_limited_workers(self, requested: int) -> int:
        """Cap the worker count so all captures in flight fit in memory.

        Every worker holds a full capture while it works. With 45 MP drone
        photos that's ~1.9 GB each, so 16 workers on a 24 GB machine exhaust
        memory before a single mask is written.
        """
        if requested <= 1:
            return requested
        memory = physical_memory_bytes()
        samples = self._capture_samples()
        if not memory or not samples:
            return requested
        per_capture = samples * _PEAK_BYTES_PER_SAMPLE
        allowed = max(1, int(memory * _MEMORY_BUDGET_FRACTION // per_capture))
        if allowed >= requested:
            return requested
        logger.warning(
            f"Using {allowed} workers instead of {requested}: each capture needs about "
            f"{per_capture / 2**30:.1f} GB while it's processed, and {requested} at once would exceed "
            f"{_MEMORY_BUDGET_FRACTION:.0%} of this machine's {memory / 2**30:.0f} GB."
        )
        return allowed

    # noinspection SpellCheckingInspection
    def process_unthreaded(
        self,
        callback: Callable[[list[str]], None] = lambda _: None,
        err_callback: Callable[[list[str], Exception], None] = lambda _s, _e: None,
    ) -> None:
        """Process all the images within the main process."""
        cur = None
        try:
            for paths in self.image_loader.paths:
                cur = paths
                self._process_one(paths)
                callback(paths)

        except Exception as exc:
            err_callback(cur, exc)
            return

    def process(
        self,
        max_workers: int = os.cpu_count(),
        callback: Callable[[list[str]], None] = lambda _: None,
        err_callback: Callable[[list[str], Exception], None] = lambda _s, _e: None,
    ) -> None:
        """Compute all glint masks.

        Computes masks for all images in self.img_paths using the process_func and save
        to the mask_out_paths.

        Parameters
        ----------
        max_workers
            The maximum number of image processing workers.
            Useful for limiting memory usage.
        callback
            Callback function passed the name of each input and output mask
            files after processing it. Will receive img_path: str as arg.
        err_callback
            Callback function passed exception object on processing failure.
            Will receive img_path: str, and the Exception as args.

        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paths = {executor.submit(self._process_one, paths): paths for paths in self.image_loader.paths}
            for future in concurrent.futures.as_completed(future_to_paths):
                paths = future_to_paths[future]
                try:
                    future.result()
                    callback(paths)

                except Exception as exc:
                    err_callback(paths, exc)
                    executor.shutdown(wait=False)
                    return

    def _process_one(self, paths: list[str] | str) -> None:
        """Generate and saves a glint mask for the image located at img_path.

        Saves the generated mask to all path locations returned by
        self.get_mask_save_paths(img_path).

        Parameters
        ----------
        paths
            The file paths used to create the image. Can be single file path or list of
            path to multiple files

        """
        img = self.image_loader.load_image(paths)
        band_scales = self.image_loader.read_band_scales(paths)
        radiometric_metadata = self.image_loader.read_radiometric_metadata(paths)
        if radiometric_metadata is not None and self.irradiance_calibrator is not None:
            radiometric_metadata = self.irradiance_calibrator.apply(radiometric_metadata)

        if self.band_aligner is not None:
            img = self.band_aligner.align(img)

        # Flag clipping on the raw DN, before preprocessing folds in exposure
        # time and irradiance and hides which pixels were actually at full well.
        saturated = img >= self.saturation_dn if self.saturation_dn is not None else None

        img = self.image_preprocessor(img, radiometric_metadata=radiometric_metadata)
        mask = self.algorithm(img, band_scales=band_scales, saturated=saturated)
        mask = self.postprocess_mask(mask)

        # Shift masks back to original unaligned coordinates for each band
        # This works for both union masks (2D) and per-band masks (3D)
        if self.band_aligner is not None:
            mask = self.band_aligner.unalign_mask(mask)

        mask = self.to_metashape_mask(mask)

        # Save per-band if we have a 3D mask (from alignment) or per_band mode
        save_per_band = mask.ndim == 3 or self.per_band  # noqa: PLR2004
        self.image_loader.save_masks(mask, paths, per_band=save_per_band)
