from .color_parse import _yaml_to_binned_colors, _yaml_to_color_reference_bin
from ..color import ColorRanges

import astropy.units as u
from astropy.coordinates import SkyCoord

from dataclasses import dataclass


SKY_CENTER_KEY = 'sky_center'

SHAPE_KEY = 'sdim'

FOV_KEY = 'fov'
FOV_UNIT_KEY = 'fov_unit'
FOV_UNIT_DEFAULT = 'arcsec'

ROTATION_KEY = 'rotation'
ROTATION_DEFAULT = 0.
ROTATION_UNIT_KEY = 'rotation_unit'
ROTATION_UNIT_DEFAULT = 'deg'

FRAME_KEY = 'frame'
FRAME_DEFAULT = 'icrs'


@dataclass
class GridModel:
    shape: tuple[int, int]
    fov: tuple[u.Quantity, u.Quantity]
    rotation: u.Quantity
    coordinate_system: str
    sky_center: SkyCoord

    color_ranges: ColorRanges
    color_reference_bin: int


def yaml_to_grid_model(grid_config: dict) -> GridModel:
    '''
    Builds the reconstruction grid from the given configuration.

    The reconstruction grid is defined by the world location, field of view
    (FOV), shape (resolution), and rotation, all specified in the input
    configuration. These parameters are extracted from the grid_config dictionary
    using helper functions.

    Parameters
    ----------
    grid_config : dict
        The configuration dictionary containing the following keys:
        - `sky_center`: World coordinate of the spatial grid center.
        - `fov`: Field of view of the grid in appropriate units.
        - `fov_unit`: (Optional) unit for the fov.
        - `sdim`: Shape of the grid, i.e. resolution, as (sdim, sdim).
        - `rotation`: Rotation of the grid.
        - `rotation_unit`: (Optional) unit for the rotation.
        - `energy_bin`: Holding `e_min`, `e_max`, and `reference_bin`.
        - `energy_unit`: The units for `e_min` and `e_max`

    '''
    sky_center = _yaml_to_sky_center(grid_config)
    shape = _yaml_to_shape(grid_config)
    fov = _yaml_to_fov(grid_config)
    rotation = _yaml_to_rotation(grid_config)
    coordinate_system = _yaml_to_coordinate_system(grid_config)

    color_ranges = _yaml_to_binned_colors(grid_config)
    color_reference_bin = _yaml_to_color_reference_bin(grid_config)

    return GridModel(
        shape=shape,
        fov=fov,
        rotation=rotation,
        coordinate_system=coordinate_system,
        sky_center=sky_center,

        color_ranges=color_ranges,
        color_reference_bin=color_reference_bin
    )


def _yaml_to_sky_center(grid_config: dict) -> SkyCoord:
    RA_KEY = 'ra'
    DEC_KEY = 'dec'

    UNIT_KEY = 'unit'
    UNIT_DEFAULT = 'deg'

    ra = grid_config[SKY_CENTER_KEY][RA_KEY]
    dec = grid_config[SKY_CENTER_KEY][DEC_KEY]
    unit = getattr(u, grid_config[SKY_CENTER_KEY].get(UNIT_KEY, UNIT_DEFAULT))
    frame = _yaml_to_coordinate_system(grid_config)

    return SkyCoord(
        ra=ra*unit,
        dec=dec*unit,
        frame=frame
    )


def _yaml_to_shape(grid_config: dict) -> tuple[int, int]:
    """Get the shape from the grid_config."""
    npix = grid_config[SHAPE_KEY]
    return (npix, npix)


def _yaml_to_fov(grid_config: dict) -> u.Quantity:
    """Get the fov from the grid_config."""

    fov = grid_config[FOV_KEY]
    unit = getattr(u, grid_config.get(FOV_UNIT_KEY, FOV_UNIT_DEFAULT))
    return (fov*unit, ) * 2


def _yaml_to_rotation(grid_config: dict) -> u.Quantity:
    """Get the rotation from the grid_config."""
    rotation = grid_config.get(ROTATION_KEY, ROTATION_DEFAULT)
    unit = getattr(u, grid_config.get(
        ROTATION_UNIT_KEY, ROTATION_UNIT_DEFAULT))
    return rotation*unit


def _yaml_to_coordinate_system(grid_config: dict) -> str:
    return grid_config.get(FRAME_KEY, FRAME_DEFAULT)
