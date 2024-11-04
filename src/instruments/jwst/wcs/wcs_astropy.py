# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

import numpy as np

from numpy.typing import ArrayLike
from typing import List, Union, Optional

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.units import Unit
from astropy import units

from .wcs_base import WcsBase

from dataclasses import dataclass


class WcsAstropy(WcsBase):
    """
    A wrapper around the astropy wcs, in order to define a common interface
    with the gwcs.
    """

    def __init__(self, header: dict):
        wcs = WCS(header)
        super().__init__(wcs)

    def wl_from_index(
        self, index: ArrayLike
    ) -> Union[SkyCoord, List[SkyCoord]]:
        """
        Convert pixel coordinates to world coordinates.

        Parameters
        ----------
        index : ArrayLike
            Pixel coordinates in the data grid.

        Returns
        -------
        wl : SkyCoord

        Note
        ----
        We use the convention of x aligning with the columns, second dimension,
        and y aligning with the rows, first dimension.
        """
        if len(np.shape(index)) == 1:
            index = [index]
        return [self._wcs.pixel_to_world(*idx) for idx in index]
        # return [self._wcs.array_index_to_world(*idx) for idx in index]

    def index_from_wl(self, wl_array: List[SkyCoord]) -> ArrayLike:
        """
        Convert world coordinates to pixel coordinates.

        Parameters
        ----------
        wl : SkyCoord

        Returns
        -------
        index : ArrayLike

        Note
        ----
        We use the convention of x aligning with the columns, second dimension,
        and y aligning with the rows, first dimension.
        """
        if isinstance(wl_array, SkyCoord):
            wl_array = [wl_array]
        return np.array([self._wcs.world_to_pixel(wl) for wl in wl_array])


@dataclass
class CoordSystemConfig:
    """Configuration for a coordinate system."""
    ctypes: tuple[str, str]
    radesys: str
    default_equinox: Optional[float] = None


# Mapping of coordinate systems to their configurations
COORD_CONFIGS = {
    'icrs': CoordSystemConfig(
        ctypes=('RA---TAN', 'DEC--TAN'),
        radesys='ICRS'
    ),
    'fk5': CoordSystemConfig(
        ctypes=('RA---TAN', 'DEC--TAN'),
        radesys='FK5',
        default_equinox=2000.0
    ),
    'fk4': CoordSystemConfig(
        ctypes=('RA---TAN', 'DEC--TAN'),
        radesys='FK4',
        default_equinox=1950.0
    ),
    'galactic': CoordSystemConfig(
        ctypes=('GLON-TAN', 'GLAT-TAN'),
        radesys='GALACTIC'
    )
}


def _check_if_implemented(coordinate_system: str):
    if coordinate_system not in COORD_CONFIGS:
        raise ValueError(f"Unsupported coordinate system: {coordinate_system}."
                         f"Supported systems {COORD_CONFIGS.keys()}")


def build_astropy_wcs(
    center: SkyCoord,
    shape: tuple[int, int],
    fov: tuple[Unit, Unit],
    rotation: Unit = 0.0 * units.deg,
    coordinate_system: str = 'icrs',
    equinox: Optional[float] = None
) -> WcsAstropy:
    """
    Create a WCS object using a FITS header.
    Build a FITS header dictionary for WCS.

    Parameters
    ----------
    center : SkyCoord
        The value of the center of the coordinate system (crval).
    shape : tuple
        The shape of the grid.
    fov : tuple
        The field of view of the grid. Typically given in degrees.
    rotation : units.Quantity
        The rotation of the grid WCS, in degrees.
    coordinate_system : str
        Coordinate system to use ('icrs', 'fk5', 'fk4', 'galactic')
    equinox : float, optional
        Equinox for FK4/FK5 systems (e.g., 2000.0 for J2000)
    """
    # Calculate rotation matrix
    rotation_value = rotation.to(units.rad).value
    pc11 = np.cos(rotation_value)
    pc12 = -np.sin(rotation_value)
    pc21 = np.sin(rotation_value)
    pc22 = np.cos(rotation_value)

    # Get coordinate system configuration
    coordinate_system = coordinate_system.lower()
    _check_if_implemented(coordinate_system)
    config = COORD_CONFIGS[coordinate_system]

    # Transform center coordinates if necessary
    if coordinate_system == 'galactic':
        lon = center.galactic.l.deg
        lat = center.galactic.b.deg
    else:
        lon = center.ra.deg
        lat = center.dec.deg

    # Build the header dictionary
    header = {
        'WCSAXES': 2,
        'CTYPE1': config.ctypes[0],
        'CTYPE2': config.ctypes[1],
        'CRPIX1': shape[0] / 2 + 0.5,
        'CRPIX2': shape[1] / 2 + 0.5,
        'CRVAL1': lon,
        'CRVAL2': lat,
        'CDELT1': -fov[0].to(units.deg).value / shape[0],
        'CDELT2': fov[1].to(units.deg).value / shape[1],
        'PC1_1': pc11,
        'PC1_2': pc12,
        'PC2_1': pc21,
        'PC2_2': pc22,
        'RADESYS': config.radesys,
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
    }

    # Set equinox if needed for FK4/FK5
    if coordinate_system in ['fk4', 'fk5']:
        header['EQUINOX'] = (
            config.default_equinox if equinox is None else equinox)

    return WcsAstropy(header)
