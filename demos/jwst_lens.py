import yaml
import argparse
import os

from functools import reduce

import nifty8.re as jft
from jax import random
import jax.numpy as jnp

import numpy as np
from matplotlib.colors import LogNorm
from astropy import units as u

import jubik0 as ju
from jubik0.likelihood import (
    connect_likelihood_to_model, build_gaussian_likelihood)
from jubik0.instruments.jwst.jwst_data import JwstData
from jubik0.instruments.jwst.masking import get_mask_from_index_centers
from jubik0.instruments.jwst.config_handler import (
    build_reconstruction_grid_from_config,
    build_coordinates_correction_prior_from_config,
    build_filter_zero_flux,
    insert_spaces_in_lensing,
    get_psf_extension_from_config,
)
from jubik0.instruments.jwst.wcs import subsample_grid_centers_in_index_grid

from jubik0.instruments.jwst.jwst_response import build_jwst_response
from jubik0.instruments.jwst.jwst_plotting import (
    build_plot_sky_residuals,
    build_plot_source,
    build_color_components_plotting,
    build_plot_lens_system, get_alpha_nonpar,
)
from jubik0.instruments.jwst.filter_projector import FilterProjector
from jubik0.instruments.jwst.pretrain_model import pretrain_lens_system

from jubik0.instruments.jwst.color import Color, ColorRange

from charm_lensing.lens_system import build_lens_system

import matplotlib.pyplot as plt
from sys import exit


filter_ranges = {
    'F2100W': ColorRange(Color(0.054*u.eV), Color(0.067*u.eV)),
    'F1800W': ColorRange(Color(0.068*u.eV), Color(0.075*u.eV)),
    'F1500W': ColorRange(Color(0.075*u.eV), Color(0.092*u.eV)),
    'F1280W': ColorRange(Color(0.093*u.eV), Color(0.107*u.eV)),
    'F1000W': ColorRange(Color(0.114*u.eV), Color(0.137*u.eV)),
    'F770W':  ColorRange(Color(0.143*u.eV), Color(0.188*u.eV)),
    'F560W':  ColorRange(Color(0.201*u.eV), Color(0.245*u.eV)),
    'F444W':  ColorRange(Color(0.249*u.eV), Color(0.319*u.eV)),
    'F356W':  ColorRange(Color(0.319*u.eV), Color(0.395*u.eV)),
    'F277W':  ColorRange(Color(0.396*u.eV), Color(0.512*u.eV)),
    'F200W':  ColorRange(Color(0.557*u.eV), Color(0.707*u.eV)),
    'F150W':  ColorRange(Color(0.743*u.eV), Color(0.932*u.eV)),
    'F115W':  ColorRange(Color(0.967*u.eV), Color(1.224*u.eV)),
}


def get_filter(color):
    # works since the filter_ranges don't overlap
    for f, cr in filter_ranges.items():
        if color in cr:
            return f


parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    help="Config File",
    type=str,
    nargs='?',
    const=1,
    default='./demos/jwst_lens_config.yaml')
args = parser.parse_args()
config_path = args.config

cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
RES_DIR = cfg['files']['res_dir']
os.makedirs(RES_DIR, exist_ok=True)
ju.save_local_packages_hashes_to_txt(
    ['nifty8', 'charm_lensing', 'jubik0'],
    os.path.join(RES_DIR, 'hashes.txt'))
ju.save_config_copy_easy(config_path, os.path.join(RES_DIR, 'config.yaml'))

if cfg['cpu']:
    from jax import config, devices
    config.update('jax_default_device', devices('cpu')[0])


reconstruction_grid = build_reconstruction_grid_from_config(cfg)
# insert_ubik_energy_in_lensing(cfg, zsource=4.2)
insert_spaces_in_lensing(cfg)
lens_system = build_lens_system(cfg['lensing'])
if cfg['nonparametric_lens']:
    sky_model = lens_system.get_forward_model_full()
else:
    sky_model = lens_system.get_forward_model_parametric()

energy_cfg = cfg['grid']['energy_bin']
e_unit = getattr(u, energy_cfg.get('unit', 'eV'))
keys_and_colors = {}
for emin, emax in zip(energy_cfg.get('e_min'), energy_cfg.get('e_max')):
    assert emin < emax
    cr = ColorRange(Color(emin*e_unit), Color(emax*e_unit))
    key = get_filter(cr.center)
    keys_and_colors[key] = cr


filter_projector = FilterProjector(
    sky_domain=sky_model.target,
    keys_and_colors=keys_and_colors,
)
sky_model_with_keys = jft.Model(
    lambda x: filter_projector(sky_model(x)),
    init=sky_model.init
)


data_dict = {}
likelihoods = []
for fltname, flt in cfg['files']['filter'].items():
    for ii, filepath in enumerate(flt):
        print(fltname, ii, filepath)
        jwst_data = JwstData(filepath)
        # print(fltname, jwst_data.half_power_wavelength)

        ekey = filter_projector.get_key(jwst_data.pivot_wavelength)
        data_key = f'{fltname}_{ekey}_{ii}'

        # Loading data, std, and mask.
        psf_ext = get_psf_extension_from_config(
            cfg, jwst_data, reconstruction_grid)

        mask = get_mask_from_index_centers(
            np.squeeze(subsample_grid_centers_in_index_grid(
                reconstruction_grid.world_extrema(ext=psf_ext),
                jwst_data.wcs,
                reconstruction_grid.wcs,
                1)),
            reconstruction_grid.shape)
        mask *= jwst_data.nan_inside_extrema(
            reconstruction_grid.world_extrema(ext=psf_ext))
        data = jwst_data.data_inside_extrema(
            reconstruction_grid.world_extrema(ext=psf_ext))
        std = jwst_data.std_inside_extrema(
            reconstruction_grid.world_extrema(ext=psf_ext))

        jwst_response = build_jwst_response(
            {ekey: sky_model_with_keys.target[ekey]},
            subsample=cfg['telescope']['rotation_and_shift']['subsample'],

            rotation_and_shift_kwargs=dict(
                reconstruction_grid=reconstruction_grid,
                data_dvol=jwst_data.dvol,
                data_wcs=jwst_data.wcs,
                data_model_type=cfg['telescope']['rotation_and_shift']['model'],
                kwargs_linear=cfg['telescope']['rotation_and_shift']['kwargs_linear'],
                world_extrema=reconstruction_grid.world_extrema(ext=psf_ext),
                shift_and_rotation_correction=dict(
                    domain_key=data_key + '_correction',
                    priors=build_coordinates_correction_prior_from_config(
                        cfg, jwst_data.filter, ii),
                )
            ),

            psf_kwargs=dict(
                camera=jwst_data.camera,
                filter=jwst_data.filter,
                center_pixel=jwst_data.wcs.index_from_wl(
                    reconstruction_grid.center)[0],
                webbpsf_path=cfg['telescope']['psf']['webbpsf_path'],
                psf_library_path=cfg['telescope']['psf']['psf_library'],
                psf_pixels=cfg['telescope']['psf'].get('psf_pixels'),
                psf_arcsec=cfg['telescope']['psf'].get('psf_arcsec'),
            ),

            transmission=jwst_data.transmission,

            data_mask=mask,

            zero_flux=dict(
                dkey=data_key,
                zero_flux=build_filter_zero_flux(cfg, jwst_data.filter),
            ),
        )

        data_dict[data_key] = dict(
            index=filter_projector.keys_and_index[ekey],
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


model = sky_model_with_keys
likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood = connect_likelihood_to_model(likelihood, model)

# PLOTTING
parametric_flag = lens_system.lens_plane_model.convergence_model.nonparametric() is not None
ll_alpha, ll_nonpar, sl_alpha, sl_nonpar = get_alpha_nonpar(lens_system)

plot_lens = build_plot_lens_system(
    RES_DIR,
    plotting_config=dict(
        norm_source=LogNorm,
        norm_lens=LogNorm,
        # norm_source_alpha=LogNorm,
        norm_source_nonparametric=LogNorm,
        # norm_mass=LogNorm,
    ),
    lens_system=lens_system,
    filter_projector=filter_projector,
    lens_light_alpha_nonparametric=(ll_alpha, ll_nonpar),
    source_light_alpha_nonparametric=(sl_alpha, sl_nonpar),
)

plot_residual = build_plot_sky_residuals(
    results_directory=RES_DIR,
    filter_projector=filter_projector,
    data_dict=data_dict,
    sky_model_with_key=sky_model_with_keys,
    small_sky_model=sky_model,
    plotting_config=dict(
        norm=LogNorm,
        data_config=dict(norm=LogNorm),
        display_pointing=False,
        xmax_residuals=4,
    ),
)
plot_color = build_color_components_plotting(
    lens_system.source_plane_model.light_model.nonparametric(), RES_DIR, substring='source')

plot_source = build_plot_source(
    RES_DIR,
    plotting_config=dict(
        norm_source=LogNorm,
        norm_source_parametric=LogNorm,
        norm_source_nonparametric=LogNorm,
        extent=lens_system.source_plane_model.space.extend().extent,
    ),
    filter_projector=filter_projector,
    source_light_model=lens_system.source_plane_model.light_model,
    source_light_alpha=sl_alpha,
    source_light_parametric=lens_system.source_plane_model.light_model.parametric(),
    source_light_nonparametric=sl_nonpar,
    attach_name=''
)

if cfg.get('prior_samples'):
    test_key, _ = random.split(random.PRNGKey(42), 2)

    def filter_data(datas: dict):
        filters = list()

        for kk, vv in datas.items():
            f = kk.split('_')[0]
            if f not in filters:
                filters.append(f)
                yield kk, vv

    prior_dict = {kk: vv for kk, vv in filter_data(data_dict)}
    plot_prior = build_plot_sky_residuals(
        results_directory=RES_DIR,
        data_dict=prior_dict,
        filter_projector=filter_projector,
        sky_model_with_key=sky_model_with_keys,
        small_sky_model=sky_model,
        plotting_config=dict(
            norm=LogNorm,
            data_config=dict(norm=LogNorm),
            display_chi2=False,
            display_pointing=False,
            std_relative=False,
        )
    )

    nsamples = cfg.get('prior_samples') if cfg.get('prior_samples') else 3
    for ii in range(nsamples):
        test_key, _ = random.split(test_key, 2)
        position = likelihood.init(test_key)
        while isinstance(position, jft.Vector):
            position = position.tree

        plot_source(position)
        plot_prior(position)
        plot_color(position)
        if not cfg['lens_only']:
            plot_lens(position, None, parametric=parametric_flag)


def plot(samples: jft.Samples, state: jft.OptimizeVIState):
    print(f'Plotting: {state.nit}')
    if cfg['plot_results']:
        plot_source(samples, state)
        plot_residual(samples, state)
        plot_color(samples, state)
        # plot_lens(samples, state, parametric=parametric_flag)


pretrain_position = pretrain_lens_system(cfg, lens_system)


cfg_mini = ju.get_config(config_path)["minimization"]
n_dof = ju.get_n_constrained_dof(likelihood)
minpars = ju.MinimizationParser(cfg_mini, n_dof, verbose=False)
key = random.PRNGKey(cfg_mini.get('key', 42))
key, rec_key = random.split(key, 2)
pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))

if pretrain_position is not None:
    while isinstance(pos_init, jft.Vector):
        pos_init = pos_init.tree

    for key in pretrain_position.keys():
        pos_init[key] = pretrain_position[key]

    pos_init = jft.Vector(pos_init)


print(f'Results: {RES_DIR}')
samples, state = jft.optimize_kl(
    likelihood,
    pos_init,
    key=rec_key,
    callback=plot,
    odir=RES_DIR,
    n_total_iterations=cfg_mini['n_total_iterations'],
    n_samples=minpars.n_samples,
    sample_mode=minpars.sample_mode,
    draw_linear_kwargs=minpars.draw_linear_kwargs,
    nonlinearly_update_kwargs=minpars.nonlinearly_update_kwargs,
    kl_kwargs=minpars.kl_kwargs,
    resume=cfg_mini.get('resume', False),
)
