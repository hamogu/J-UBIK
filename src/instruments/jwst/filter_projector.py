# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %
from .grid import Grid

import nifty8.re as jft
from typing import Optional
import numpy as np


def sorted_keys_and_index(keys_and_colors: dict):
    keys, colors = keys_and_colors.keys(), keys_and_colors.values()
    sorted_indices = np.argsort([c.center.energy.value for c in colors])
    return {
        key: index for key, index in zip(keys, sorted_indices)
    }


class FilterProjector(jft.Model):
    """
    A nifty8.re.Model that projects input data into specified filters
    defined by color keys.

    The FilterProjector class takes a sky domain and a mapping between keys
    and colors, and applies a projection of input data according to the filters.
    It supports querying keys based on colors and efficiently applies
    transformations for multi-channel inputs.
    """

    def __init__(
        self,
        sky_domain: jft.ShapeWithDtype,
        keys_and_colors: dict,
        sorted: Optional[bool] = True
    ):
        """
        Parameters
        ----------
        sky_domain : jft.ShapeWithDtype
            The domain for the sky data, defining the shape and data type of
            the input.
        keys_and_colors : dict
            A dictionary where the keys are filter names (or keys) and the
            values are lists of colors associated with each filter.
            This defines how inputs will be mapped to the respective filters.
        sorted : bool
            If `True` the indices will be ordered in ascending energy.
            Corresponding to the energies of the colors in the
            `keys_and_colors` dictionary.
        """

        assert len(sky_domain.shape) == 3, ('FilterProjector expects a sky '
                                            'with 3 dimensions.')

        self.keys_and_colors = keys_and_colors
        if sorted:
            self.keys_and_index = sorted_keys_and_index(keys_and_colors)
        else:
            self.keys_and_index = {
                key: index for index, key in enumerate(keys_and_colors.keys())
            }
        super().__init__(domain=sky_domain)

    def get_key(self, color):
        """Returns the key that corresponds to the given color."""
        out_key = ''
        for k, v in self.keys_and_colors.items():
            if color in v:
                if out_key != '':
                    raise IndexError(
                        f'{color} fits into multiple keys of the '
                        'FilterProjector')
                out_key = k
        if out_key == '':
            raise IndexError(
                f"{color} doesn't fit in the bounds of the FilterProjector.")

        return out_key

    def __call__(self, x):
        return {key: x[index] for key, index in self.keys_and_index.items()}


def build_filter_projector_from_named_color_ranges(
    sky_domain: jft.ShapeWithDtype,
    grid: Grid,
    named_color_ranges: dict,
    data_filter_names: list[str],
):
    keys_and_colors = {}
    for grid_color_range in grid.spectral:
        for name in data_filter_names:
            jwst_filter = named_color_ranges[name.upper()]
            if grid_color_range.center in jwst_filter:
                keys_and_colors[name] = grid_color_range

    return FilterProjector(
        sky_domain=sky_domain,
        keys_and_colors=keys_and_colors,
    )
