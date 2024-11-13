# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %
from .grid import Grid
from ...hashcollector import save_local_packages_hashes_to_txt
from ...utils import save_config_copy_easy

import os
import yaml

from typing import Optional

from astropy import units as u


def load_yaml_and_save_info(config_path):
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
    results_directory = cfg['files']['res_dir']
    os.makedirs(results_directory, exist_ok=True)
    save_local_packages_hashes_to_txt(
        ['nifty8', 'charm_lensing', 'jubik0'],
        os.path.join(results_directory, 'hashes.txt'))
    save_config_copy_easy(config_path, os.path.join(
        results_directory, 'config.yaml'))
    return cfg, results_directory


def build_filter_zero_flux(
    config: dict,
    filter: str,
) -> dict:
    """
    Builds the zero flux prior for the specified filter.

    This function retrieves the zero flux prior for a given filter from the
    configuration dictionary. If the filter-specific prior is not available,
    it falls back to a general prior defined in the configuration.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing the zero flux priors for
        different filters under `config['telescope']['zero_flux']`.

    filter : str
        The name of the filter (case-insensitive) for which the zero flux prior
        is to be built.

    Returns
    -------
    dict
        A dictionary containing the zero flux prior for the specified filter.
        If the filter is not present, returns the default prior.
    """
    prior_config = config['telescope']['zero_flux']
    lower_filter = filter.lower()

    if lower_filter in prior_config:
        return dict(prior=prior_config[lower_filter])

    return dict(prior=prior_config['prior'])


def build_coordinates_correction_prior_from_config(
    config: dict,
    filter: Optional[str] = '',
    filter_data_set_id: Optional[int] = 0
) -> dict:
    """
    Builds the coordinate correction prior for the specified filter and dataset.

    The function extracts the shift and rotation priors for the given filter and
    dataset ID from the configuration.
    If the specific filter or dataset ID is not found, it returns the default
    shift and rotation priors. The rotation prior is converted to radians
    if needed.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing rotation and shift priors under
        `config['telescope']['rotation_and_shift']['priors']`.

    filter : Optional[str], default=''
        The name of the filter (case-insensitive) for which the prior is needed.
        If not specified, the default priors are used.

    filter_data_set_id : Optional[int], default=0
        The dataset ID for which the prior is needed. If not provided or not
        found, the function uses the default dataset priors.

    Returns
    -------
    dict
        A dictionary containing the shift and rotation priors for the specified
        filter and dataset. If the filter or dataset is not found, returns the
        default priors.
    """
    rs_priors = config['telescope']['rotation_and_shift']['priors']

    lower_filter = filter.lower()
    if ((lower_filter in rs_priors) and
            (filter_data_set_id in rs_priors.get(lower_filter, dict()))):
        shift = rs_priors[lower_filter][filter_data_set_id]['shift']
        rotation = rs_priors[lower_filter][filter_data_set_id]['rotation']

    else:
        shift = rs_priors['shift']
        rotation = rs_priors['rotation']

    rotation_unit = getattr(u, rs_priors.get('rotation_unit', 'deg'))
    rotation = (rotation[0],
                rotation[1],
                (rotation[2] * rotation_unit).to(u.rad).value)
    return dict(shift=shift, rotation=rotation)


def config_transform(config: dict):
    """
    Recursively transforms string values in a configuration dictionary.

    This function processes a dictionary and attempts to evaluate any string
    values that may represent valid Python expressions. If the string cannot
    be evaluated, it is left unchanged. The function also applies the same
    transformation recursively for any nested dictionaries.

    Parameters
    ----------
    config : dict
        The configuration dictionary where string values may be transformed.
        If a value is a string that can be evaluated, it is replaced by the
        result of `eval(val)`. Nested dictionaries are processed recursively.
    """
    for key, val in config.items():
        if isinstance(val, str):
            try:
                config[key] = eval(val)
            except:
                continue
        elif isinstance(val, dict):
            config_transform(val)


def get_grid_extension_from_config(
    config: dict,
    reconstruction_grid: Grid,
):
    '''Load the grid extension for the reconstruction grid. The reconstruction
    gets zero padded by this amount. This is needed to avoid wrapping flux due
    to fft convolution of the psf.

    Parameters
    ----------
    config: dict
        The config dict holds psf_arcsec_extension, which holds the extension
        in units of arcsec.
    reconstruction_grid: Grid
        The grid underlying the reconstruction.

    Returns
    -------
    grid_extension: tuple[int]
        A pixel number tuple, that specifies by how many pixels the
        reconstruction will be zero padded.
    '''

    psf_arcsec_extension = config['telescope']['psf'].get(
        'psf_arcsec_extension')
    if psf_arcsec_extension is None:
        raise ValueError('Need to provide either `psf_arcsec_extension`.')

    return [int((psf_arcsec_extension*u.arcsec).to(u.deg) / 2 / dist)
            for dist in reconstruction_grid.spatial.distances]


def _parse_insert_spaces(cfg):
    lens_fov = cfg['grid']['fov']
    lens_npix = cfg['grid']['sdim']
    lens_padd = cfg['grid']['s_padding_ratio']
    lens_npix = (lens_npix, lens_npix)
    lens_dist = [lens_fov/p for p in lens_npix]
    lens_energy_bin = cfg['grid']['energy_bin']
    lens_space = dict(padding_ratio=lens_padd,
                      Npix=lens_npix,
                      distance=lens_dist,
                      energy_bin=lens_energy_bin
                      )

    source_fov = cfg['grid']['source_grid']['fov']
    source_npix = cfg['grid']['source_grid']['sdim']
    source_padd = cfg['grid']['source_grid']['s_padding_ratio']
    source_npix = (source_npix, source_npix)
    source_dist = [source_fov/p for p in source_npix]
    source_energy_bin = cfg['grid']['energy_bin']
    source_space = dict(padding_ratio=source_padd,
                        Npix=source_npix,
                        distance=source_dist,
                        energy_bin=source_energy_bin,
                        )

    return lens_space, source_space


def insert_spaces_in_lensing(cfg):
    lens_space, source_space = _parse_insert_spaces(cfg)

    cfg['lensing']['spaces'] = dict(
        lens_space=lens_space, source_space=source_space)


def insert_spaces_in_lensing_new(cfg):
    lens_space, source_space = _parse_insert_spaces(cfg)

    cfg['spaces'] = dict(
        lens_space=lens_space, source_space=source_space)
