from ..color import ColorRanges, ColorRange, Color
from astropy import units


ENERGY_UNIT_KEY = 'energy_unit'
ENERGY_BIN_KEY = 'energy_bin'
REFERENCE_BIN_KEY = 'reference_bin'


def yaml_to_binned_colors(grid_config: dict) -> ColorRanges:
    EMIN_KEY = 'e_min'
    EMAX_KEY = 'e_max'

    color_ranges = []
    emins = grid_config[ENERGY_BIN_KEY][EMIN_KEY]
    emaxs = grid_config[ENERGY_BIN_KEY][EMAX_KEY]
    eunit = getattr(units, grid_config[ENERGY_UNIT_KEY])
    for emin, emax in zip(emins, emaxs):
        emin, emax = emin*eunit, emax*eunit
        color_ranges.append(ColorRange(Color(emin), Color(emax)))

    return ColorRanges(color_ranges)


def yaml_to_color_reference_bin(grid_config: dict) -> int:
    return grid_config[ENERGY_BIN_KEY][REFERENCE_BIN_KEY]
