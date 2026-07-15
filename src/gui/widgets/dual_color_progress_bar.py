"""Progress bar that keeps its percentage text readable regardless of chunk fill.

The default QProgressBar draws the text once in a single color, so when the fill
chunk color matches the text color, the text disappears the moment the chunk
covers it. This subclass paints the text twice — once in the "on light" color
clipped to the empty region, and once in the "on dark" color clipped to the
filled region — so the digits stay visible across the whole bar.
"""

from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QProgressBar, QStyle, QStyleOptionProgressBar


class DualColorProgressBar(QProgressBar):
    """QProgressBar that draws its percentage text in two colors so it stays visible."""

    # Match brutalist.qss palette: navy chunk on off-white background.
    _TEXT_ON_LIGHT = QColor("#31353f")
    _TEXT_ON_DARK = QColor("#f4f5f4")

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt method)
        option = QStyleOptionProgressBar()
        self.initStyleOption(option)

        painter = QPainter(self)
        try:
            style = self.style()
            # Draw the frame + chunk, but let us handle the label ourselves.
            option.textVisible = False
            style.drawControl(QStyle.ControlElement.CE_ProgressBar, option, painter, self)

            if not self.isTextVisible():
                return

            text = self.text()
            if not text:
                return

            # Compute the pixel x where the chunk ends inside the drawable rect.
            groove = style.subElementRect(
                QStyle.SubElement.SE_ProgressBarGroove, option, self
            )
            span = self.maximum() - self.minimum()
            fraction = 0.0 if span <= 0 else (self.value() - self.minimum()) / span
            fraction = max(0.0, min(1.0, fraction))
            chunk_end_x = groove.left() + int(round(groove.width() * fraction))

            label_rect = style.subElementRect(
                QStyle.SubElement.SE_ProgressBarLabel, option, self
            )

            # Draw text once in the "on light" color, clipped to the unfilled region.
            painter.save()
            painter.setClipRect(QRect(chunk_end_x, 0, self.width() - chunk_end_x, self.height()))
            painter.setPen(self._TEXT_ON_LIGHT)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)
            painter.restore()

            # And once in the "on dark" color, clipped to the filled region.
            painter.save()
            painter.setClipRect(QRect(0, 0, chunk_end_x, self.height()))
            painter.setPen(self._TEXT_ON_DARK)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)
            painter.restore()
        finally:
            painter.end()
