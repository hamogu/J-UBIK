import yaml
import argparse

from functools import reduce

import nifty8.re as jft
from jax import random
import jax.numpy as jnp

import numpy as np
from matplotlib.colors import LogNorm
from astropy import units as u

import jubik0 as ju

from charm_lensing.lens_system import build_lens_system
# from charm_lensing.physical_models.multifrequency_models.nifty_mf import build_nifty_mf_from_grid

from jubik0.instruments.jwst.filter_projector import (
    build_filter_projector_from_named_color_ranges)
from jubik0.instruments.jwst.pretrain_model import pretrain_lens_system
from jubik0.instruments.jwst.config_handler import load_yaml_and_save_info
from jubik0.instruments.jwst.config_handler import (
    insert_spaces_in_lensing_new)
from jubik0.likelihood import connect_likelihood_to_model
from jubik0.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods

from jubik0.instruments.jwst.plotting.plotting import get_plot, plot_prior

from jubik0.instruments.jwst.parse.grid import yaml_to_grid_model
from jubik0.instruments.jwst.grid import Grid
from jubik0.instruments.jwst.jwst_data import JWST_FILTERS
from jubik0.instruments.jwst.color import Color, ColorRange

from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    help="Config File",
    type=str,
    nargs='?',
    const=1,
    default='./demos/configs/spt0418.yaml')
args = parser.parse_args()
config_path = args.config

cfg, results_directory = load_yaml_and_save_info(config_path)

if cfg['cpu']:
    from jax import config, devices
    config.update('jax_default_device', devices('cpu')[0])

if cfg['no_interactive_plotting']:
    import matplotlib
    matplotlib.use('Agg')


grid_model = yaml_to_grid_model(cfg['sky']['grid'])
grid = Grid.from_grid_model(grid_model)


# insert_ubik_energy_in_lensing(cfg, zsource=4.2)
insert_spaces_in_lensing_new(cfg['sky'])
lens_system = build_lens_system(cfg['sky'])
if cfg['nonparametric_lens']:
    sky_model = lens_system.get_forward_model_full()
    parametric_flag = False
else:
    sky_model = lens_system.get_forward_model_parametric()
    parametric_flag = True

# # For testing
# sky_model = build_nifty_mf_from_grid(
#     grid,
#     'test',
#     cfg['sky']['model']['source']['light']['multifrequency']['nifty_mf'],
#     reference_bin=grid_model.color_reference_bin,
# )


named_color_ranges = {}
for name, values in JWST_FILTERS.items():
    pivot, bw, er, blue, red = values
    named_color_ranges[name] = ColorRange(Color(red*u.um), Color(blue*u.um))

filter_projector = build_filter_projector_from_named_color_ranges(
    sky_domain=sky_model.target,
    grid=grid,
    named_color_ranges=named_color_ranges,
    data_filter_names=cfg['files']['filter'].keys()
)
for fpt, fpc in filter_projector.target.items():
    print(fpt, fpc)

sky_model_with_keys = jft.Model(
    lambda x: filter_projector(sky_model(x)),
    init=sky_model.init
)

likelihoods, data_dict = build_jwst_likelihoods(
    cfg, grid, filter_projector, sky_model_with_keys)

likelihood = reduce(lambda x, y: x+y, likelihoods)
likelihood = connect_likelihood_to_model(likelihood, sky_model_with_keys)

plot_source, plot_residual, plot_color, plot_lens = get_plot(
    results_directory, lens_system, filter_projector, data_dict, sky_model,
    sky_model_with_keys, cfg, parametric_flag)
if cfg.get('prior_samples'):
    plot_prior(
        cfg, likelihood, filter_projector, sky_model, sky_model_with_keys,
        plot_source, plot_lens, plot_color, data_dict, parametric_flag)


def plot(samples: jft.Samples, state: jft.OptimizeVIState):
    print(f'Plotting: {state.nit}')
    if cfg['plot_results']:
        plot_source(samples, state)
        plot_residual(samples, state)
        plot_color(samples, state)
        plot_lens(samples, state, parametric=parametric_flag)


cfg_mini = ju.get_config(config_path)["minimization"]
n_dof = ju.get_n_constrained_dof(likelihood)
minpars = ju.MinimizationParser(cfg_mini, n_dof, verbose=False)
key = random.PRNGKey(cfg_mini.get('key', 42))
key, rec_key = random.split(key, 2)
pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))

pretrain_position = pretrain_lens_system(cfg, lens_system)
if pretrain_position is not None:
    while isinstance(pos_init, jft.Vector):
        pos_init = pos_init.tree

    for key in pretrain_position.keys():
        pos_init[key] = pretrain_position[key]

    pos_init = jft.Vector(pos_init)


print(f'Results: {results_directory}')
samples, state = jft.optimize_kl(
    likelihood,
    pos_init,
    key=rec_key,
    callback=plot,
    odir=results_directory,
    n_total_iterations=cfg_mini['n_total_iterations'],
    n_samples=minpars.n_samples,
    sample_mode=minpars.sample_mode,
    draw_linear_kwargs=minpars.draw_linear_kwargs,
    nonlinearly_update_kwargs=minpars.nonlinearly_update_kwargs,
    kl_kwargs=minpars.kl_kwargs,
    resume=cfg_mini.get('resume', False),
)
