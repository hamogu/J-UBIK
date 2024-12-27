import yaml
import argparse

import nifty8.re as jft

import numpy as np
from matplotlib.colors import LogNorm

from charm_lensing.lens_system import build_lens_system

from jubik0.instruments.jwst.config_handler import (
    insert_spaces_in_lensing_new)
from jubik0.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods

from jubik0.instruments.jwst.parse.grid import yaml_to_grid_model
from jubik0.instruments.jwst.grid import Grid

from os.path import join
from sys import exit

from jax import config, devices
config.update('jax_default_device', devices('cpu')[0])


def plot_radial_light_distribution():
    import matplotlib.pyplot as plt

    from jubik0.instruments.jwst.plotting.jwst_plotting import (
        _get_data_model_and_chi2, _get_model_samples_or_position,
        _determine_ypos)
    import astropy.units as u

    def get_pixel_radius_from_max_value(sky):
        shape = sky.shape
        xx, yy = np.meshgrid(*(np.arange(shape[1]), np.arange(shape[0])))
        xmax, ymax = np.unravel_index(np.argmax(sky), sky.shape)
        return np.hypot(yy-xmax, xx-ymax)

    radii = []
    radial_fm = []
    radial_data = []
    radial_ll = []
    radial_sl = []
    zz = {'f2100w': 221.56061654,
          'f1800w': 92.48311537,
          'f1500w': 42.64578287,
          'f1280w': 24.97585328,
          'f1000w': 13.98030351,
          'f770w': 4.2036279,
          'f560w': 1.22458536,
          'f444w': 0.14007522,
          'f356w': 0.01034513,
          'f277w': 0.01760561,
          'f200w': 0.02736996,
          'f150w': 0.00221736,
          'f115w': 0.11671709}

    slm = lens_system.get_forward_model_parametric(only_source=True)
    sky_sl_with_keys = jft.Model(
        lambda x: filter_projector({'sky': slm(x)}),
        init=slm.init
    )
    llm = lens_system.lens_plane_model.light_model
    sky_ll_with_keys = jft.Model(
        lambda x: filter_projector({'sky': llm(x)}),
        init=llm.init
    )
    sky_model_with_keys = jft.Model(
        lambda x: filter_projector(sky_model(x)),
        init=sky_model.init
    )

    sky = _get_model_samples_or_position(
        samples, sky_model_with_keys)
    sky_sl = _get_model_samples_or_position(
        samples, sky_sl_with_keys)
    sky_ll = _get_model_samples_or_position(
        samples, sky_ll_with_keys)

    for dkey, dval in data_dict.items():
        flt, _ = dkey.split('_')
        ddist = np.sqrt(dval['data_dvol']).to(u.arcsec).value
        print(flt, ddist)

        data_model = dval['data_model']
        # to_brightness = (
        #     1/(dval['data_dvol'] * dval['data_transmission'])).value
        data = dval['data']  # * to_brightness

        full_model_mean = _get_data_model_and_chi2(
            samples,
            sky,
            data_model=data_model,
            data=data,
            mask=dval['mask'],
            std=dval['std'])[0]  # * to_brightness
        ll_model_mean = _get_data_model_and_chi2(
            samples,
            sky_ll,
            data_model=data_model,
            data=data,
            mask=dval['mask'],
            std=dval['std'])[0]  # * to_brightness
        sl_model_mean = _get_data_model_and_chi2(
            samples,
            sky_sl,
            data_model=data_model,
            data=data,
            mask=dval['mask'],
            std=dval['std'])[0]  # * to_brightness

        rel_r = get_pixel_radius_from_max_value(ll_model_mean)
        max = int(np.ceil(np.max(rel_r)))
        pixel_radius = np.linspace(0, np.max(rel_r), max)
        ddist = np.sqrt(dval['data_dvol']).to(u.arcsec).value

        pr = []  # physical radius
        fm_radial = []
        ll_radial = []
        sl_radial = []
        data_radial = []
        for ii in range(pixel_radius.shape[0]-1):
            mask = ((pixel_radius[ii] < rel_r) *
                    (rel_r < pixel_radius[ii+1]) *
                    dval['mask'])
            pr.append(ii*ddist)
            fm_radial.append(np.nanmean(full_model_mean[mask]))
            ll_radial.append(np.nanmean(ll_model_mean[mask]))
            sl_radial.append(np.nanmean(sl_model_mean[mask]))
            data_radial.append(np.nanmean(data[mask]))
        radii.append(np.array(pr))
        radial_fm.append(np.array(fm_radial))
        radial_ll.append(np.array(ll_radial))
        radial_sl.append(np.array(sl_radial))
        radial_data.append(np.array(data_radial))

    xlen = len(sky_model_with_keys.target)
    fig, axes = plt.subplots(3, xlen, figsize=(3*xlen, 8), dpi=300)
    axes = axes.T
    vmin, vmax = 0.01, 100
    dmin, dmax = -0.2, 2.0
    jj_prev = 1e9
    for dkey, pr,  data_radial, fm_radial, ll_radial, sl_radial in zip(
            data_dict.keys(), radii, radial_data, radial_fm, radial_ll, radial_sl):
        flt, _ = dkey.split('_')
        jj = _determine_ypos(dkey, filter_projector, 0)
        ax = axes[jj]
        if jj != jj_prev:
            ax[0].set_title(flt)
            ax[0].plot(pr, data_radial-zz[flt], label='data', color='black')
            ax[0].plot(pr, fm_radial-zz[flt], label='full_model')
            ax[0].plot(pr, ll_radial-zz[flt], label='lens_light_model')
            ax[0].plot(pr, sl_radial-zz[flt], label='source_light_model')
            ax[0].loglog()
            ax[1].plot(pr, data_radial-fm_radial, label='data - full_model')
            ax[1].plot(pr, data_radial-ll_radial,
                       label='data - lens_light_model')
            ax[1].plot(pr, data_radial-sl_radial,
                       label='data - source_light_model')
            if jj == 0:
                ax[0].legend()
                ax[1].legend()
        else:
            ax[2].plot(pr, data_radial-fm_radial)
            ax[2].plot(pr, data_radial-ll_radial)
            ax[2].plot(pr, data_radial-sl_radial)

        ax[0].set_ylim(bottom=vmin, top=vmax)
        ax[1].set_ylim(bottom=dmin, top=dmax)
        ax[2].set_ylim(bottom=dmin, top=dmax)

        jj_prev = jj

    plt.tight_layout()
    plt.savefig(join(results_directory, 'radial_profile.png'))
    plt.close()


def get_sersic_parameters_for_copy_model():
    lens_lights = lens_system.lens_plane_model.light_model.nonparametric.components
    sersic = {
        'ie': [[] for _ in range(len(lens_lights))],
        're': [[] for _ in range(len(lens_lights))],
        'n': [[] for _ in range(len(lens_lights))],
        'c': [[] for _ in range(len(lens_lights))],
        't': [[] for _ in range(len(lens_lights))],
        'q': [[] for _ in range(len(lens_lights))],
    }
    for s in samples:
        for ii, ll in enumerate(lens_lights):
            ie, re, n, c, t, q = ll.parametric.prior(s)
            sersic['ie'][ii].append(ie)
            sersic['re'][ii].append(re)
            sersic['n'][ii].append(n)
            sersic['c'][ii].append(np.hypot(*c))
            sersic['t'][ii].append(t)
            sersic['q'][ii].append(q)
    sersic = {k: np.array(v) for k, v in sersic.items()}
    sersic_means = {k: np.mean(v, axis=1) for k, v in sersic.items()}


def get_filter_shifts_and_rotations():
    rotation_models = {k: v['data_model'].rotation_and_shift
                       for k, v in data_dict.items()}
    shifts = {k: jft.mean([v.coordinates.shift_prior(s) for s in samples])
              for k, v in rotation_models.items()}
    rotations = {k: jft.mean([v.coordinates.rotation_prior(s) for s in samples])
                 for k, v in rotation_models.items()}
    return shifts, rotations


def load_samples(odir):
    import os
    import pickle
    last_fn = os.path.join(odir, "last.pkl")
    with open(last_fn, "rb") as f:
        samples, opt_vi_st = pickle.load(f)
    return samples


SKY_KEY = 'sky'

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


plot_radial_light_distribution()
