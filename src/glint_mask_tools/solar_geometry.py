"""Solar position and DLS sun-angle correction.

Implements MicaSense's documented DLS (Downwelling Light Sensor) correction
model, ported from the ``micasense.dls`` module of the public
`imageprocessing <https://github.com/micasense/imageprocessing>`_ library.

Why this exists: the DLS is rigidly mounted to the aircraft body, so its
sensing plane tilts with aircraft attitude. A simple "project onto horizontal"
correction that assumes the sun is directly overhead (``E / (cos(pitch) *
cos(roll))``) is a reasonable approximation only when aircraft tilt is small
or the sun actually is near zenith. At higher tilt (e.g., a UAV pitching
significantly to hold airspeed into a headwind) the error grows because the
correction ignores where the sun actually is relative to the direction the
aircraft is pointed (its heading/yaw) — tilting toward the sun and tilting
away from it require different corrections, which a pitch/roll-only model
cannot distinguish.

The proper model:
  1. Compute the sun's position (elevation, azimuth) from GPS location and UTC
     timestamp via a standard low-precision solar position algorithm
     (Meeus/NOAA formulas, accurate to a fraction of a degree).
  2. Compute the DLS sensor's pointing direction in world (NED) coordinates by
     rotating its body-frame "up" vector through the aircraft's yaw, pitch,
     and roll.
  3. The angle between the sun vector and the sensor-pointing vector — the
     "sun-sensor angle" — is the actual angle of incidence on the DLS's
     diffuser dome, correctly accounting for tilt direction relative to the
     sun, not just tilt magnitude.
  4. Correct for Fresnel transmission loss through the dome at that angle.
  5. Split the corrected reading into direct + diffuse components (using an
     assumed clear-sky direct:diffuse ratio) and project the direct component
     onto a horizontal plane using the sun's elevation — this is the
     irradiance a level, upward-facing sensor would have measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

# Clear-sky direct:diffuse irradiance ratio. MicaSense's own tooling uses ~6:1
# as a reasonable default absent a per-flight calibration measurement.
_DEFAULT_DIRECT_TO_DIFFUSE_RATIO = 6.0

# DLS sensor's "up" direction in its own body frame, expressed in NED
# (North-East-Down): the Down component is -1, i.e. the dome points away from
# Down (up), matching MicaSense's ``dls_orientation_vector = [0, 0, -1]``.
_DLS_ORIENTATION_NED = np.array([0.0, 0.0, -1.0])

# Refractive indices for the DLS dome's air -> polycarbonate-ish dome -> diffuser
# stack, from MicaSense's fresnel() default: n=[1.000277, 1.6, 1.38].
_DOME_REFRACTIVE_INDICES = (1.000277, 1.6, 1.38)


@dataclass(frozen=True)
class SunPosition:
    """Sun position at a given place and time."""

    elevation_rad: float
    azimuth_rad: float  # compass bearing, 0 = north, clockwise positive


def solar_position(latitude_deg: float, longitude_deg: float, dt_utc: datetime) -> SunPosition:
    """Compute solar elevation and azimuth using the NOAA/Meeus low-precision algorithm.

    Parameters
    ----------
    latitude_deg
        Observer latitude, decimal degrees, positive north.
    longitude_deg
        Observer longitude, decimal degrees, positive east (negative west).
    dt_utc
        Capture time as a timezone-aware (or naive-assumed-UTC) datetime in UTC.

    Returns
    -------
    SunPosition with elevation and azimuth in radians. Azimuth is a compass
    bearing (0=N, 90=E, 180=S, 270=W).

    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    # Julian day (fractional), then Julian century from J2000.0.
    jd = _julian_day(dt_utc)
    t = (jd - 2451545.0) / 36525.0

    # Geometric mean longitude and anomaly of the sun (degrees).
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    m_rad = math.radians(m)

    # Eccentricity of Earth's orbit.
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    # Sun's equation of center.
    c = (
        math.sin(m_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * m_rad) * (0.019993 - 0.000101 * t)
        + math.sin(3 * m_rad) * 0.000289
    )

    true_long = l0 + c
    apparent_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * t))

    # Mean + corrected obliquity of the ecliptic.
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * t))
    eps_rad = math.radians(eps)
    lambda_rad = math.radians(apparent_long)

    # Declination.
    declination_rad = math.asin(math.sin(eps_rad) * math.sin(lambda_rad))

    # Equation of time (minutes).
    y = math.tan(eps_rad / 2.0) ** 2
    l0_rad = math.radians(l0)
    eq_time = 4.0 * math.degrees(
        y * math.sin(2 * l0_rad)
        - 2 * e * math.sin(m_rad)
        + 4 * e * y * math.sin(m_rad) * math.cos(2 * l0_rad)
        - 0.5 * y * y * math.sin(4 * l0_rad)
        - 1.25 * e * e * math.sin(2 * m_rad)
    )

    # True solar time (minutes) directly in UTC: longitude contributes
    # 4 minutes per degree east; timezone offset is 0 since we work in UTC.
    utc_minutes = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0 + dt_utc.microsecond / 6.0e7
    true_solar_time = (utc_minutes + eq_time + 4.0 * longitude_deg) % 1440.0

    hour_angle_deg = true_solar_time / 4.0 - 180.0
    if true_solar_time < 0:
        hour_angle_deg += 360.0
    hour_angle_rad = math.radians(hour_angle_deg)

    lat_rad = math.radians(latitude_deg)
    cos_zenith = math.sin(lat_rad) * math.sin(declination_rad) + math.cos(lat_rad) * math.cos(
        declination_rad
    ) * math.cos(hour_angle_rad)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith_rad = math.acos(cos_zenith)
    elevation_rad = math.pi / 2.0 - zenith_rad

    # Azimuth (compass bearing, clockwise from north).
    sin_zenith = math.sin(zenith_rad)
    if abs(sin_zenith) < 1e-6:
        azimuth_rad = 0.0
    else:
        cos_az = (
            math.sin(lat_rad) * cos_zenith - math.sin(declination_rad)
        ) / (math.cos(lat_rad) * sin_zenith)
        cos_az = max(-1.0, min(1.0, cos_az))
        az_deg = math.degrees(math.acos(cos_az))
        azimuth_deg = (az_deg + 180.0) % 360.0 if hour_angle_deg > 0 else (540.0 - az_deg) % 360.0
        azimuth_rad = math.radians(azimuth_deg)

    return SunPosition(elevation_rad=elevation_rad, azimuth_rad=azimuth_rad)


def _julian_day(dt_utc: datetime) -> float:
    """Fractional Julian day for a UTC datetime (standard algorithm)."""
    y, mo = dt_utc.year, dt_utc.month
    d = dt_utc.day + (dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0) / 24.0
    if mo <= 2:  # noqa: PLR2004
        y -= 1
        mo += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (mo + 1)) + d + b - 1524.5


def sun_vector_ned(sun: SunPosition) -> np.ndarray:
    """Unit vector pointing toward the sun, in NED (North-East-Down) coordinates."""
    cos_el = math.cos(sun.elevation_rad)
    return np.array(
        [
            cos_el * math.cos(sun.azimuth_rad),
            cos_el * math.sin(sun.azimuth_rad),
            -math.sin(sun.elevation_rad),
        ]
    )


def sensor_orientation_ned(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    """DLS sensor pointing direction in NED, given aircraft yaw/pitch/roll.

    Ported verbatim (sign conventions included) from MicaSense's
    ``dls.get_orientation``. Rotates the sensor's body-frame "up" vector
    ``[0, 0, -1]`` through yaw, then pitch, then roll to get its true pointing
    direction in the world (NED) frame.
    """
    c1, s1 = math.cos(-yaw_rad), math.sin(-yaw_rad)
    c2, s2 = math.cos(-pitch_rad), math.sin(-pitch_rad)
    c3, s3 = math.cos(-roll_rad), math.sin(-roll_rad)
    r_yaw = np.array([[c1, s1, 0.0], [-s1, c1, 0.0], [0.0, 0.0, 1.0]])
    r_pitch = np.array([[c2, 0.0, -s2], [0.0, 1.0, 0.0], [s2, 0.0, c2]])
    r_roll = np.array([[1.0, 0.0, 0.0], [0.0, c3, s3], [0.0, -s3, c3]])
    r = r_yaw @ (r_pitch @ r_roll)
    return r @ _DLS_ORIENTATION_NED


def sun_sensor_angle(
    sun: SunPosition,
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
) -> float:
    """Angle (radians) between the sun direction and the DLS sensor's pointing direction."""
    n_sun = sun_vector_ned(sun)
    n_sensor = sensor_orientation_ned(yaw_rad, pitch_rad, roll_rad)
    cos_angle = float(np.dot(n_sun, n_sensor))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle)


def _fresnel_transmission(phi: float, n1: float, n2: float) -> float:
    """Unpolarized Fresnel transmittance at incidence angle phi (radians)."""
    f1 = math.cos(phi)
    inner = 1.0 - (n1 / n2 * math.sin(phi)) ** 2
    if inner < 0:
        return 0.0
    f2 = math.sqrt(inner)
    rs = ((n1 * f1 - n2 * f2) / (n1 * f1 + n2 * f2)) ** 2
    rp = ((n1 * f2 - n2 * f1) / (n1 * f2 + n2 * f1)) ** 2
    t = 1.0 - 0.5 * rs - 0.5 * rp
    if not (0.0 <= t <= 1.0):
        return 0.0
    return t


def dome_transmission(phi: float) -> float:
    """Multilayer Fresnel transmittance through the DLS dome stack at angle phi."""
    n = _DOME_REFRACTIVE_INDICES
    t = 1.0
    phi_eff = phi
    for i in range(len(n) - 1):
        n1, n2 = n[i], n[i + 1]
        sin_inner = math.sin(phi_eff) / n1
        sin_inner = max(-1.0, min(1.0, sin_inner))
        phi_eff = math.asin(sin_inner)
        t *= _fresnel_transmission(phi_eff, n1, n2)
    return t


def horizontal_irradiance(
    measured_irradiance: float,
    sun: SunPosition,
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
    direct_to_diffuse_ratio: float = _DEFAULT_DIRECT_TO_DIFFUSE_RATIO,
) -> float:
    """Correct a raw DLS irradiance reading to what a level, upward sensor would read.

    Parameters
    ----------
    measured_irradiance
        Raw DLS irradiance reading (any consistent unit; output is in the same unit).
    sun
        Sun position at capture time (see ``solar_position``).
    yaw_rad, pitch_rad, roll_rad
        Aircraft/DLS attitude at capture time, radians.
    direct_to_diffuse_ratio
        Assumed clear-sky ratio of direct to diffuse irradiance. Default 6.0
        matches MicaSense's typical clear-sky assumption absent a calibrated
        per-flight estimate.

    Returns
    -------
    Irradiance corrected to a horizontal (nadir-up) sensing plane.

    """
    angle = sun_sensor_angle(sun, yaw_rad, pitch_rad, roll_rad)
    angular_correction = dome_transmission(angle)
    if angular_correction <= 1e-6:  # noqa: PLR2004
        # Extreme incidence angle (sensor nearly edge-on to the sun); fall back
        # to an uncorrected reading rather than dividing by ~0.
        angular_correction = 1.0

    sensor_irradiance = measured_irradiance / angular_correction
    percent_diffuse = 1.0 / direct_to_diffuse_ratio
    denom = percent_diffuse + math.cos(angle)
    if denom <= 1e-6:  # noqa: PLR2004
        # Sun-sensor angle near 90°+: direct component contribution is
        # ill-conditioned. Treat as fully diffuse rather than blow up.
        return sensor_irradiance * percent_diffuse
    untilted_direct = sensor_irradiance / denom
    scattered = untilted_direct * percent_diffuse
    return untilted_direct * math.sin(sun.elevation_rad) + scattered
