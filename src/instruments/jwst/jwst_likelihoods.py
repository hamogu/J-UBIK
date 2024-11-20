from ...likelihood import (build_gaussian_likelihood)
from .config_handler import (get_grid_extension_from_config,)

from .jwst_response import build_jwst_response
from .jwst_data import (load_jwst_data_mask_std)
from .filter_projector import FilterProjector
from .grid import Grid

# Parsing
from .parse.jwst_psf import (yaml_to_psf_kernel_config)
from .parse.zero_flux_model import (yaml_to_zero_flux_prior_config)
from .parse.rotation_and_shift.coordinates_correction import (
    yaml_to_coordinates_correction_config)
from .parse.rotation_and_shift.rotation_and_shift import (
    yaml_to_rotation_and_shift_algorithm_config)
from .jwst_psf import load_psf_kernel_from_config


import jax.numpy as jnp
import nifty8.re as jft


def build_jwst_likelihoods(
    cfg: dict,
    grid: Grid,
    filter_projector: FilterProjector,
    sky_model_with_keys: jft.Model,
):
    '''Build the likelihoods according to the '''

    zero_flux_prior_configs = yaml_to_zero_flux_prior_config(
        cfg['telescope']['zero_flux'])
    psf_kernel_configs = yaml_to_psf_kernel_config(cfg['telescope']['psf'])
    coordiantes_correction_config = yaml_to_coordinates_correction_config(
        cfg['telescope']['rotation_and_shift']['correction_priors'])
    rotation_and_shift_algorithm_config = yaml_to_rotation_and_shift_algorithm_config(
        cfg['telescope']['rotation_and_shift'])

    data_dict = {}
    likelihoods = []
    for fltname, flt in cfg['files']['filter'].items():
        for ii, filepath in enumerate(flt):
            print(fltname, ii, filepath)

            # Loading data, std, and mask.
            grid_extension = get_grid_extension_from_config(cfg, grid)
            world_corners = grid.spatial.world_extrema(ext=grid_extension)

            jwst_data, data, mask, std = load_jwst_data_mask_std(
                filepath, grid, world_corners)

            data_subsample = cfg['telescope']['rotation_and_shift']['subsample']
            psf_kernel = load_psf_kernel_from_config(
                jwst_data=jwst_data,
                pointing_center=grid.spatial.center,
                subsample=data_subsample,
                config_parameters=psf_kernel_configs,
            )

            energy_name = filter_projector.get_key(jwst_data.pivot_wavelength)
            data_identifier = f'{fltname}_{ii}'

            jwst_response = build_jwst_response(
                sky_domain={
                    energy_name: sky_model_with_keys.target[energy_name]},
                data_identifier=data_identifier,
                data_subsample=data_subsample,

                rotation_and_shift_kwargs=dict(
                    reconstruction_grid=grid,
                    data_dvol=jwst_data.dvol,
                    data_wcs=jwst_data.wcs,
                    algorithm_config=rotation_and_shift_algorithm_config,
                    world_extrema=world_corners,
                ),
                shift_and_rotation_correction_prior=coordiantes_correction_config.get_filter_or_default(
                    jwst_data.filter, ii),

                psf_kernel=psf_kernel,
                transmission=jwst_data.transmission,
                zero_flux_prior_config=zero_flux_prior_configs.get_filter_or_default(
                    fltname),
                data_mask=mask,
            )

            data_dict[data_identifier] = dict(
                index=filter_projector.keys_and_index[energy_name],
                data=data,
                std=std,
                mask=mask,
                data_model=jwst_response,
                data_dvol=jwst_data.dvol,
                data_transmission=jwst_data.transmission,
            )

            likelihood = build_gaussian_likelihood(
                jnp.array(data[mask], dtype=float),
                jnp.array(std[mask], dtype=float))
            likelihood = likelihood.amend(
                jwst_response, domain=jft.Vector(jwst_response.domain))
            likelihoods.append(likelihood)

    return likelihoods, data_dict
