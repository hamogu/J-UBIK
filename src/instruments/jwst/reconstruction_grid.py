# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

from typing import Tuple, Optional

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from numpy.typing import ArrayLike

from .wcs.wcs_astropy import build_astropy_wcs
from .color import BinnedColorRanges


class Grid:
    """
    Grid that wraps a 2D array with a world coordinate system (WCS).

    This class represents a grid in the sky with a world coordinate system
    (WCS) and centered around a given sky location.
    It provides methods to calculate physical properties of the grid,
    such as spatial_distances between pixels, and allows easy access to the world and
    relative coordinates of the grid points.
    """

    def __init__(
        self,
        spatial_center: SkyCoord,
        spatial_shape: Tuple[int, int],
        spatial_fov: Tuple[Unit, Unit],
        spectral_colors: BinnedColorRanges,
        spatial_rotation: Unit = 0.0 * units.deg,
        spatial_coordinate_system: Optional[str] = 'icrs',
    ):
        """
        Initialize the Grid with a spatial_center, spatial_shape, spatial 
        field of view, spectral_colors, and optional spatial_rotation.

        Parameters
        ----------
        spatial_center : SkyCoord
            The central sky coordinate of the grid.
        spatial_shape : tuple of int
            The spatial_shape of the grid, specified as (rows, columns).
        spatial_fov : tuple of Unit
            The field of view of the grid in angular units for both axes
            (width, height).
        spectral_colors: BinnedColorRanges
            The spectral_colors of the energy dimension.
        spatial_rotation : Unit, optional
            The spatial_rotation of the grid in degrees, counterclockwise from
            north. Default is 0 degrees.
        spatial_coordinate_system: Optional[str] = 'icrs',
            The coordinate system of the spatial domain.
        """

        # Create spatial coordinates
        self.spatial_shape = spatial_shape
        self.spatial_fov = spatial_fov
        self.spatial_distances = [
            f.to(units.deg)/s for f, s in zip(spatial_fov, spatial_shape)]
        self.spatial_center = spatial_center
        self.spatial_wcs = build_astropy_wcs(
            center=spatial_center,
            shape=spatial_shape,
            fov=(spatial_fov[0].to(units.deg),
                 spatial_fov[1].to(units.deg)),
            rotation=spatial_rotation,
            coordinate_system=spatial_coordinate_system,
        )

        self.spectral_colors = spectral_colors

    @property
    def shape(self):
        '''Shape of the grid. (spectral, spatial)'''
        return self.spectral_colors.shape + self.spatial_shape

    @property
    def spatial_dvol(self) -> Unit:
        """Computes the area of a grid cell (pixel) in angular units."""
        return self.spatial_distances[0] * self.spatial_distances[1]

    def spatial_world_extrema(
        self,
        extend_factor: float = 1,
        ext: Optional[tuple[int, int]] = None
    ) -> ArrayLike:
        """
        The world location of the spatial_center of the pixels with the index
        locations = ((0, 0), (0, -1), (-1, 0), (-1, -1))

        Parameters
        ----------
        extend_factor : float, optional
            A factor by which to extend the grid. Default is 1.
        ext : tuple of int, optional
            Specific extension values for the grid's rows and columns.

        Returns
        -------
        ArrayLike
            The world coordinates of the corner pixels.

        Note
        ----
        The indices are assumed to coincide with the convention of the first
        index (x) aligning with the columns and the second index (y) aligning
        with the rows.
        """
        if ext is None:
            ext0, ext1 = [int(shp*extend_factor-shp) //
                          2 for shp in self.spatial_shape]
        else:
            ext0, ext1 = ext

        xmin = -ext0
        xmax = self.spatial_shape[0] + ext1  # - 1 FIXME: Which of the two
        ymin = -ext1
        ymax = self.spatial_shape[1] + ext1  # - 1
        return self.spatial_wcs.wl_from_index([
            (xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)])

    def spatial_index_grid(
        self,
        extend_factor=1,
        to_bottom_left=True
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute the grid of indices for the array.

        Parameters
        ----------
        extend_factor : float, optional
            A factor to increase the grid size. Default is 1 (no extension).
        to_bottom_left : bool, optional
            Whether to shift the indices of the extended array such that (0, 0)
            is aligned with the upper left corner of the unextended array.
            Default is True.

        Returns
        -------
        tuple of ArrayLike
            The meshgrid of index arrays for the extended grid.

        Example
        -------
            un_extended = (0, 1, 2)
            extended_centered = (-1, 0, 1, 2, 3)
            extended_bottom_left = (0, 1, 2, 3, -1)
        """
        extent = [int(s * extend_factor) for s in self.spatial_shape]
        extent = [(e - s) // 2 for s, e in zip(self.spatial_shape, extent)]
        x, y = [np.arange(-e, s+e) for e, s in zip(extent, self.spatial_shape)]
        if to_bottom_left:
            x, y = np.roll(x, -extent[0]), np.roll(y, -extent[1])
        return np.meshgrid(x, y, indexing='xy')

    def spatial_distances_in_units_of(self, unit: Unit) -> list[float]:
        return [d.to(unit).value for d in self.spatial_distances]

    def extent(self, unit=units.arcsec):
        """Convenience method which gives the extent of the grid in
        physical units."""
        spatial_distances = [d.to(unit).value for d in self.spatial_distances]
        halfside = np.array(self.spatial_shape)/2 * np.array(spatial_distances)
        return -halfside[0], halfside[0], -halfside[1], halfside[1]
