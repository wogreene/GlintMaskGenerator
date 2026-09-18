"""CLI functions for the Glint Masker Generator.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-05-30
Description: Command line interface to the glint-mask-tools.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Callable

import typer
from tqdm import tqdm

from .sensors import Sensor, _known_sensors

if TYPE_CHECKING:
    from .maskers import Masker


app = typer.Typer()


def _err_callback(paths: list[str], exception: Exception) -> None:
    tqdm.write(f"{paths} failed with err:\n{exception}", file=sys.stderr)


def _process(masker: Masker, max_workers: int) -> None:
    with tqdm(total=len(masker)) as progress:
        masker(
            max_workers=max_workers,
            callback=lambda _: progress.update(1),
            err_callback=_err_callback,
        )


def _create_sensor_command(sensor_cfg: Sensor) -> Callable[..., None]:
    """Create a CLI command function for a sensor configuration."""

    def sensor_command(  # noqa: PLR0913
        img_dir: Annotated[
            Path,
            typer.Argument(
                exists=True,
                file_okay=True,
                dir_okay=True,
                help="The path to a named input image or directory containing images. "
                "If img_dir is a directory, all tif, jpg, jpeg, and png images in that directory will be processed.",
            ),
        ],
        out_dir: Annotated[
            Path,
            typer.Argument(
                exists=True,
                file_okay=True,
                dir_okay=True,
                help='The path to send your out image including the file name and type. e.g. "/path/to/mask.png". '
                "The out_dir must be a directory if img_dir is specified as a directory.",
            ),
        ],
        thresholds: Annotated[
            list[float],
            typer.Option(
                help="The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0).",
            ),
        ] = tuple(b.default_threshold for b in sensor_cfg.bands),
        pixel_buffer: Annotated[
            int,
            typer.Option(help="The pixel distance to buffer out the mask."),
        ] = 0,
        max_workers: Annotated[
            int,
            typer.Option(help="The maximum number of threads to use for processing."),
        ] = min(4, os.cpu_count()),
        per_band: Annotated[  # noqa: FBT002
            bool,
            typer.Option(
                help="Mask each band independently without union. Only applies to multi-band sensors.",
            ),
        ] = False,
        no_align: Annotated[  # noqa: FBT002
            bool,
            typer.Option(
                "--no-align",
                help="Disable automatic band alignment for multi-band sensors.",
            ),
        ] = False,
        alignment: Annotated[
            str,
            typer.Option(
                "--alignment",
                help=(
                    "Alignment strategy: 'rig' (calibrated XMP metadata, MicaSense only), "
                    "'phase' (content-based phase correlation), or 'default' (sensor default)."
                ),
            ),
        ] = "default",
        benthos_index_max: Annotated[
            float,
            typer.Option(
                "--benthos-index-max",
                help=(
                    "Spare the seafloor from the mask. Only pixels whose water-column index "
                    "(RedEdge717 vs NIR842 on MicaSense) is below this value stay masked. "
                    "Water absorbs 842nm far harder than 717nm, so anything seen through water "
                    "reads high (reef/coral/sand around +0.6) while surface glint and foam read "
                    "low (+0.1 to +0.2). Typical value: 0.35. Set to a large negative number "
                    "(default: -100) to disable."
                ),
            ),
        ] = -100.0,
        no_irradiance_stabilization: Annotated[  # noqa: FBT002
            bool,
            typer.Option(
                "--no-irradiance-stabilization",
                help=(
                    "Disable the flight-level DLS irradiance model for MicaSense sensors. "
                    "By default, captures whose sun-sensor geometry makes their own DLS reading "
                    "unusable (aircraft tilted away from a low sun) have their irradiance "
                    "interpolated from captures in the same flight that do have usable geometry."
                ),
            ),
        ] = False,
        contrast_multiplier: Annotated[
            float,
            typer.Option(
                "--contrast-multiplier",
                help=(
                    "Also mask pixels brighter than this multiple of the capture's own background "
                    "level in the sensor's glint reference band (NIR where there is one). Works on "
                    "captures where sensor clipping compresses the reflectance scale and a fixed "
                    "threshold can't be reached. Around 3 tracks visible glint on water imagery. "
                    "Values <= 1 disable it (default)."
                ),
            ),
        ] = 0.0,
        whitewash: Annotated[  # noqa: FBT002
            bool,
            typer.Option(
                "--whitewash",
                help=(
                    "RGB only: replace the per-band thresholds with the colour + local-contrast "
                    "whitewash detector. Masks colourless pixels that are very bright, or fairly "
                    "bright and brighter than their surroundings; spares turquoise sand, pale reef "
                    "flats and orange benthos."
                ),
            ),
        ] = False,
        whitewash_bright_floor: Annotated[
            float, typer.Option(help="Whitewash detector: dimmest-channel value (0-1) masked outright.")
        ] = 0.80,
        whitewash_contrast_floor: Annotated[
            float,
            typer.Option(help="Whitewash detector: dimmest-channel value (0-1) needed before local contrast applies."),
        ] = 0.65,
        whitewash_local_contrast: Annotated[
            float,
            typer.Option(help="Whitewash detector: how many times brighter than its surroundings a pixel must be."),
        ] = 1.35,
        no_spare_orange: Annotated[  # noqa: FBT002
            bool,
            typer.Option("--no-spare-orange", help="Whitewash detector: don't exempt orange benthos."),
        ] = False,
        no_mask_saturated: Annotated[  # noqa: FBT002
            bool,
            typer.Option(
                "--no-mask-saturated",
                help=(
                    "Don't mask pixels that hit the sensor's full well. By default they are "
                    "masked regardless of threshold: a clipped pixel's reflectance is a floor, "
                    "not a measurement, and with a long auto-exposure that floor can fall below "
                    "any sensible threshold, dropping real glint out of the mask."
                ),
            ),
        ] = False,
    ) -> None:
        if thresholds is None:
            thresholds = sensor_cfg.get_default_thresholds()

        strategy = None if alignment == "default" else alignment
        # Sentinel: values <= -1.0 mean "disabled". Real index thresholds are in [-1, 1].
        benthos_index = benthos_index_max if benthos_index_max > -1.0 else None
        masker = sensor_cfg.create_masker(
            str(img_dir),
            str(out_dir),
            thresholds,
            pixel_buffer,
            per_band=per_band,
            align_bands=not no_align,
            alignment_strategy=strategy,
            benthos_index_max=benthos_index,
            stabilize_irradiance=not no_irradiance_stabilization,
            mask_saturated=not no_mask_saturated,
            # A multiplier of 1 or less would flag most of the frame, so it doubles as "off".
            contrast_multiplier=contrast_multiplier if contrast_multiplier > 1.0 else None,
            whitewash=(
                {
                    "bright_floor": whitewash_bright_floor,
                    "contrast_floor": whitewash_contrast_floor,
                    "local_contrast": whitewash_local_contrast,
                    "spare_orange": not no_spare_orange,
                }
                if whitewash
                else None
            ),
        )
        _process(masker, max_workers)

    sensor_command.__doc__ = f"Generate glint masks for {sensor_cfg.name} sensors using threshold algorithm."
    return sensor_command


# Dynamically register sensor commands
for cfg in _known_sensors:
    command_func = _create_sensor_command(cfg.sensor)
    app.command(name=cfg.cli_name)(command_func)


if __name__ == "__main__":
    app()
