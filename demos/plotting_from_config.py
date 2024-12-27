import yaml
import argparse
import os

import pickle

import nifty8.re as jft

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, FuncNorm
from astropy import units as u

from jubik0.instruments.jwst.jwst_data import JwstData
from jubik0.instruments.jwst.masking import get_mask_from_index_centers
from jubik0.instruments.jwst.config_handler import (
    build_reconstruction_grid_from_config,
    build_coordinates_correction_prior_from_config,
    build_filter_zero_flux,
    insert_spaces_in_lensing)
from jubik0.instruments.jwst.wcs import (subsample_grid_centers_in_index_grid)
# from jubik0.instruments.jwst.jwst_data_model import build_data_model
from jubik0.instruments.jwst.jwst_plotting import (
    build_plot_sky_residuals,
    build_plot_lens_system,
    build_plot_source,
    get_alpha_nonpar,
    rgb_plotting,
)
from jubik0.instruments.jwst.filter_projector import FilterProjector

from jubik0.instruments.jwst.color import Color, ColorRange

from charm_lensing.lens_system import build_lens_system

from sys import exit

from jax import config, devices
config.update('jax_default_device', devices('cpu')[0])


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


PLOT_RESIDUALS = False
PLOT_ALL_RECONSTRUCTION = False

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


reconstruction_grid = build_reconstruction_grid_from_config(cfg)
# insert_ubik_energy_in_lensing(cfg, zsource=4.2)
insert_spaces_in_lensing(cfg)
lens_system = build_lens_system(cfg['lensing'])
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


parametric_flag = lens_system.lens_plane_model.convergence_model.nonparametric() is not None
ll_alpha, ll_nonpar, sl_alpha, sl_nonpar = get_alpha_nonpar(lens_system)

source_light = lens_system.source_plane_model.light_model
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

cfm = lens_system.source_plane_model.light_model.nonparametric()
plot_deviations = build_plot_source(
    RES_DIR,
    plotting_config=dict(
        # norm_source=LogNorm,
        norm_source_parametric=LogNorm,
        norm_source_nonparametric=LogNorm,
        extent=lens_system.source_plane_model.space.extend().extent,
    ),
    filter_projector=filter_projector,
    source_light_model=cfm.spectral_deviations_distribution,
    # source_light_model=cfm.spectral_distribution,
    source_light_alpha=sl_alpha,
    source_light_parametric=lens_system.source_plane_model.light_model.parametric(),
    source_light_nonparametric=sl_nonpar,
    attach_name='dev_'
)


if PLOT_ALL_RECONSTRUCTION:
    n_iters = cfg['minimization']['n_total_iterations']
    for ii in range(n_iters):
        # if ii < 35:
        #     continue

        samp_stat = os.path.join(RES_DIR, f'position_{ii: 02d}')
        if not os.path.isfile(samp_stat):
            samp_stat = os.path.join(RES_DIR, f'samples_state_{ii:02d}.pkl')

        if os.path.isfile(samp_stat):
            with open(samp_stat, "rb") as f:
                samples, opt_vi_st = pickle.load(f)

            lens_path = os.path.join(RES_DIR, 'lens', f'{ii:02d}.png')
            res_path = os.path.join(RES_DIR, 'residuals', f'{ii:02d}.png')
            source_path = os.path.join(RES_DIR, 'source', f'{ii:02d}.png')
            dev_path = os.path.join(RES_DIR, 'source', f'dev_{ii:02d}.png')

            if not os.path.isfile(source_path):
                print('Plotting source', ii)
                plot_source(samples, opt_vi_st)

            if not os.path.isfile(dev_path) and ii > 45:
                print('Plotting source deviations', ii)
                plot_deviations(samples, opt_vi_st)

elif False:
    samp_stat = os.path.join(RES_DIR, 'last.pkl')
    with open(samp_stat, "rb") as f:
        samples, opt_vi_st = pickle.load(f)
    plot_source(samples, opt_vi_st)
    plot_deviations(samples, opt_vi_st)

else:
    samp_stat = os.path.join(RES_DIR, 'last.pkl')
    with open(samp_stat, "rb") as f:
        samples, opt_vi_st = pickle.load(f)

    from charm_lensing.plotting import (
        configure_matplotlib, display_colorbar, configer_panel, display_text)
    COLUMN_WIDTH, TEXT_WIDTH, FONT_PROPERTIES = configure_matplotlib(
        widths='aanda')

    source_light_model = lens_system.source_plane_model.light_model

    source_light = jft.mean([source_light_model(x) for x in samples])

    xlen = 5
    ylen = 3
    min_source = 1.0e-3

    fig, axes = plt.subplots(ylen, xlen, figsize=(2*xlen, 1.8*ylen), dpi=300)
    ims = np.zeros_like(axes)
    axes = axes.flatten()
    ims = ims.flatten()
    for ii, (fltname, fld) in enumerate(filter_projector(source_light).items()):
        # axes[ii].set_title(f'{fltname}')
        ims[ii] = axes[ii].imshow(
            fld,
            origin='lower',
            norm=LogNorm(vmin=min_source, vmax=source_light.max()),
            cmap='jet',
        )
        configer_panel(axes[ii], ims[ii], cbar_setting=dict(display=False))
        display_text(axes[ii], f'{fltname}')

    for ax in [axes[-2], axes[-1]]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect('equal')

    bar = plt.colorbar(ims[ii], cax=axes[-2])
    bar.set_label('MJy/sr')

    fig.tight_layout()
    fig.savefig('tmp.pdf')
    plt.close()


exit()
if __name__ == "__main__":

    rgb_plotting(
        lens_system, samples, three_filter_names=('f1000w', 'f770w', 'f560w')
    )

    llm = lens_system.lens_plane_model.light_model
    for comp, fltname in zip(
            llm.nonparametric().components, cfg['files']['filter'].keys()):
        print(fltname)
        distro = comp.parametric().prior
        prior_keys = comp.parametric().model.prior_keys
        ms, ss = jft.mean_and_std([distro(s) for s in samples])
        for (k, _), m, s in zip(prior_keys, ms, ss):
            print(k, m, s, sep='\t')
        print()

    sm = lens_system.source_plane_model.light_model
    sm_non = sm.nonparametric()

    from jubik0.jwst.jwst_plotting import (_get_data_model_and_chi2,
                                           _get_sky_or_skies)
    sky = _get_sky_or_skies(samples, sky_model_with_keys)

    llm = lens_system.lens_plane_model.light_model
    sky_ll_with_keys = jft.Model(
        lambda x: filter_projector(llm(x)),
        init=llm.init
    )
    sky_ll = _get_sky_or_skies(samples, sky_ll_with_keys)

    slm = lens_system.get_forward_model_parametric(only_source=True)
    sky_sl_with_keys = jft.Model(
        lambda x: filter_projector(slm(x)),
        init=slm.init
    )
    sky_sl = _get_sky_or_skies(samples, sky_sl_with_keys)
