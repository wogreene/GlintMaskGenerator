"""Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17.
"""

from PyQt6 import QtWidgets, uic

from gui.utils import resource_path

# Tooltips differ by threshold scale: DN-normalized sensors (RGB, CIR, P4MS, M3M)
# use the full 0-1 range, while MicaSense sensors work in surface reflectance where
# even strong glint rarely exceeds 0.08.
_DN_TOOLTIP = (
    "Brightness threshold as a fraction of the sensor's full range (0-1), where 1.0 is a "
    "saturated pixel. Pixels at or above this value in every thresholded band are masked. "
    "Lower value = more pixels masked. Defaults sit near the top of the range because only "
    "near-saturated pixels are glint on 8-bit imagery."
)
_REFLECTANCE_TOOLTIP = (
    "Surface reflectance threshold. 1.0 would be 100% reflectance (perfect Lambertian "
    "diffuser), but on real MicaSense flight data even strong glint / wave-crest specular "
    "reflectance typically tops out around 0.03-0.08 — nowhere near 1.0. Typical values: "
    "0.02-0.03 catches broad glint + bright wave chop; 0.04-0.06 isolates stronger glint only. "
    "Tune per-flight — favorable-attitude passes read brighter than oblique/tilted passes for "
    "the same physical glint. Type a value above the slider's range in the box if needed."
)


class ThresholdCtrl(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, max_value: float = 1.0) -> None:
        super().__init__(parent)

        uic.loadUi(resource_path("resources/threshold_ctrl.ui"), self)

        # Scale the slider to the sensor's useful range. The spinbox always allows the
        # full 0-1 so a value beyond the slider's range can still be typed in.
        self.slider.setMaximum(int(max_value * 1000))
        tooltip = _DN_TOOLTIP if max_value > 0.5 else _REFLECTANCE_TOOLTIP  # noqa: PLR2004
        self.slider.setToolTip(tooltip)
        self.spinbox.setToolTip(tooltip)

        self.slider.valueChanged.connect(
            lambda value: self.spinbox.setValue(value / 1000.0),
        )
        self.spinbox.valueChanged.connect(
            lambda value: self.slider.setValue(int(value * 1000)),
        )

        self.show()

    @property
    def value(self) -> float:
        return self.spinbox.value()

    @value.setter
    def value(self, v: float) -> None:
        self.slider.setValue(int(v * 1000))
        self.spinbox.setValue(v)
