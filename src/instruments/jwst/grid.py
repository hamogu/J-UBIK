# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

import numpy as np


from .wcs.wcs_astropy import WcsAstropy
from .color import ColorRanges

from .parse.grid_parse import GridModel


class Grid:
    """
    Grid provides the physical coordinates for a sky brightness model.

    The class holds a spatial coordiante system, which includes
        - `spatial` world coordinate system (WcsAstropy), which locates the
          spatial grid in astrophysical coordinates.
        - `spectral` coordinate system (ColorRanges), which provides a spectral
          coordinate range to the frequency/energy/wavelength bins of the sky
          brightness model.
        - `polarization_labels`, for now fixed to Stokes I.
        - `times`, for now fixed to eternity.
    """

    def __init__(
        self,
        spatial: WcsAstropy,
        spectral: ColorRanges,
    ):
        """
        Initialize the Grid with a `spatial` and `spectral` coordinate system.

        Parameters
        ----------
        spatial : WcsAstropy
            The spatial coordinate system.
        spectral: ColorRanges
            The spectral coordinate system.
        """

        # Spatial
        self.spatial = spatial
        # Spectral
        self.spectral = spectral
        # Polarization, TODO: Implement more options.
        self.polarization_labels = ['I']
        # Time, TODO: Implement more options
        self.times = [-np.inf, np.inf]

    @classmethod
    def from_grid_model(cls, grid_model: GridModel):
        spatial = WcsAstropy(
            center=grid_model.sky_center,
            shape=grid_model.shape,
            fov=grid_model.fov,
            rotation=grid_model.rotation,
            coordinate_system=grid_model.coordinate_system,
        )

        spectral = grid_model.color_ranges

        return Grid(spatial, spectral)

    @property
    def shape(self):
        '''Shape of the grid. (spectral, spatial)'''
        return self.spectral.shape + self.spatial.shape
