import yaml
import argparse

import nifty8.re as jft
from jax import random
import jax.numpy as jnp

import numpy as np
from matplotlib.colors import LogNorm

import jubik0 as ju

from charm_lensing.lens_system import build_lens_system
# from charm_lensing.physical_models.multifrequency_models.nifty_mf import build_nifty_mf_from_grid

from jubik0.instruments.jwst.pretrain_model import pretrain_lens_system
from jubik0.instruments.jwst.config_handler import load_yaml_and_save_info
from jubik0.instruments.jwst.config_handler import (
    insert_spaces_in_lensing_new)
from jubik0.likelihood import connect_likelihood_to_model
from jubik0.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods

from jubik0.instruments.jwst.plotting.plotting import get_plot, plot_prior

from jubik0.instruments.jwst.jwst_likelihoods import build_filter_projector

from jubik0.instruments.jwst.parse.grid import yaml_to_grid_model
from jubik0.instruments.jwst.grid import Grid
from jubik0.instruments.jwst.rotation_and_shift.coordinates_correction import (
    build_coordinates_correction_from_grid)

from sys import exit

from jax import config, devices
config.update('jax_default_device', devices('cpu')[0])

SKY_KEY = 'sky'


def load_samples(odir):
    import os
    import pickle
    last_fn = os.path.join(odir, "last.pkl")
    with open(last_fn, "rb") as f:
        samples, opt_vi_st = pickle.load(f)
    return samples


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

cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
results_directory = cfg['files']['res_dir']

grid = Grid.from_grid_model(yaml_to_grid_model(cfg['sky']['grid']))

# insert_ubik_energy_in_lensing(cfg, zsource=4.2)
insert_spaces_in_lensing_new(cfg['sky'])
lens_system = build_lens_system(cfg['sky'])
if cfg['nonparametric_lens']:
    sky_model = lens_system.get_forward_model_full()
    parametric_flag = False
else:
    sky_model = lens_system.get_forward_model_parametric()
    parametric_flag = True
sky_model = jft.Model(jft.wrap_left(sky_model, SKY_KEY),
                      domain=sky_model.domain)

samples = load_samples(results_directory)

likelihood, filter_projector, data_dict = build_jwst_likelihoods(
    cfg, grid, sky_model, sky_key=SKY_KEY)

rotation_models = {k: v['data_model'].rotation_and_shift
                   for k, v in data_dict.items()}

shifts = {k: jft.mean([v.coordinates.shift_prior(s) for s in samples])
          for k, v in rotation_models.items()}
rotations = {k: jft.mean([v.coordinates.rotation_prior(s) for s in samples])
             for k, v in rotation_models.items()}
