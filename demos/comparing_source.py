import os
import yaml

import pickle

import nifty8.re as jft
from jax import random


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, FuncNorm, Normalize
from astropy import units as u


import jubik0 as ju
from jubik0.instruments.jwst.config_handler import (
    build_reconstruction_grid_from_config,
    insert_spaces_in_lensing,
)

from jubik0.instruments.jwst.config_handler import _get_rotation
from jubik0.instruments.jwst.filter_projector import FilterProjector


from jubik0.instruments.jwst.color import Color, ColorRange
from resolve.ubik_tools.fits import field2fits

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


def load_last_pickle(config_path: str):
    resume_fn = config_path.split('/')[:-1]
    resume_fn.append("last.pkl")
    resume_fn = '/'.join(resume_fn)

    with open(resume_fn, "rb") as f:
        samples, opt_vi_st = pickle.load(f)

    return samples, opt_vi_st


def load_reconstruction(config_path: str):
    cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
    insert_spaces_in_lensing(cfg)
    lens_system = build_lens_system(cfg['lensing'])
    sky_model = lens_system.source_plane_model.light_model

    energy_cfg = cfg['grid']['energy_bin']
    e_unit = getattr(u, energy_cfg.get('unit', 'eV'))
    keys_and_colors = {}
    for emin, emax in zip(energy_cfg.get('e_min'), energy_cfg.get('e_max')):
        assert emin < emax
        cr = ColorRange(Color(emin*e_unit), Color(emax*e_unit))
        key = get_filter(cr.center)
        keys_and_colors[key] = cr
    fp = FilterProjector(
        sky_domain=sky_model.target,
        keys_and_colors=keys_and_colors,
    )

    samples, state = load_last_pickle(config_path)

    ssamples = np.array([sky_model(s) for s in samples])
    smean, sstd = jft.mean_and_std(ssamples)
    return smean, sstd, fp, sky_model, ssamples, samples


def plot_nd_array(nd_array, norm):
    xlen, ylen = nd_array.shape[1], nd_array.shape[0]
    fig, axes = plt.subplots(ylen, xlen, sharex=True, sharey=True)
    for ii, axe in enumerate(axes):
        for jj, (fname, ax) in enumerate(zip(full_fp.keys_and_index.keys(), axe)):
            im = ax.imshow(
                nd_array[ii, jj], origin='lower', extent=source_grid.extent(),
                norm=norm)
            plt.colorbar(im, ax=ax)
    plt.show()


def get_contour(g, cval):
    return g/g.max() > cval


def get_sed(rsamples, contour):
    galaxy_vals = rsamples[..., contour]
    gal_mean = galaxy_vals.mean(axis=0).mean(axis=-1)
    gal_std = galaxy_vals.std(axis=0).mean(axis=-1)
    return gal_mean, gal_std


full_path = '/home/jruestig/pro/python/j-ubik/results/jwst_lens/mfreworked/all_data_05_biggerpsf/config.yaml'
f444w_path = '/home/jruestig/pro/python/j-ubik/results/jwst_lens/filter_consistency_test/f444w_single_01/config.yaml'
f1280w_path = '/home/jruestig/pro/python/j-ubik/results/jwst_lens/filter_consistency_test/f1280w_single_01/config.yaml'
f1500w_path = '/home/jruestig/pro/python/j-ubik/results/jwst_lens/filter_consistency_test/f1500w_single_01/config.yaml'
f1800w_path = '/home/jruestig/pro/python/j-ubik/results/jwst_lens/filter_consistency_test/f1800w_single_01/config.yaml'
f2100w_path = '/home/jruestig/pro/python/j-ubik/results/jwst_lens/filter_consistency_test/f2100w_single_01/config.yaml'


cfg = yaml.load(open(full_path, 'r'), Loader=yaml.SafeLoader)

insert_spaces_in_lensing(cfg)
lens_system = build_lens_system(cfg['lensing'])
lens_grid = build_reconstruction_grid_from_config(cfg)
source_grid = ju.Grid(
    center=lens_grid.center,
    shape=lens_system.source_plane_model.space.extend().shape,
    fov=(lens_system.source_plane_model.space.extend().fov * u.arcsec,)*2,
    rotation=_get_rotation(cfg))


full_smean, full_sstd, full_fp, full_m, full_samples, full_latent_samples = load_reconstruction(
    full_path)

# SED fitting
parametric = lens_system.source_plane_model.light_model.parametric()

galaxy_1, galaxy_2 = [jft.mean([p(fls) for fls in full_latent_samples]) for
                      p in parametric.parametric()]

cval = 0.4
contour_1, contour_2 = [get_contour(g, cval) for g in [galaxy_1, galaxy_2]]
contour_all = np.ones_like(contour_1)
sed_galaxy_1_mean, sed_galaxy_1_std = get_sed(full_samples, contour_1)
sed_galaxy_2_mean, sed_galaxy_2_std = get_sed(full_samples, contour_2)
sed_all_mean, sed_all_std = get_sed(full_samples, contour_all)
sed_x_labels = [k for k in full_fp.keys_and_colors.keys()]
sed_x_wavelength = [c.center.wavelength.to(
    u.um).value for c in full_fp.keys_and_colors.values()]

# plotting sed with source
out_path = '/'.join(full_path.split('/')[:-1] + ['posterior_analysis'])
norm = LogNorm(vmin=1e-3)
fig, axes = plt.subplots(4, 4, figsize=(12, 10), dpi=300)
for ii, (ax, fname) in enumerate(zip(axes.flatten(), full_fp.keys_and_colors.keys())):
    im = ax.imshow(full_smean[ii], origin='lower', extent=source_grid.extent(),
                   norm=norm, cmap='viridis')
    ax.set_title(fname)
    plt.colorbar(im, ax=ax)
    ax.contour(contour_1, colors='orange', linewidth=0.5,
               alpha=0.7, extent=source_grid.extent())
    ax.contour(contour_2, colors='green', linewidth=0.5,
               alpha=0.7, extent=source_grid.extent())
ax = axes[-1, -1]
ax.errorbar(sed_x_wavelength, sed_galaxy_1_mean, yerr=sed_galaxy_1_std,
            label='galaxy 1', color='orange', fmt='o', capsize=3)
ax.errorbar(sed_x_wavelength, sed_galaxy_2_mean, yerr=sed_galaxy_2_std,
            label='galaxy 2', color='green', fmt='o', capsize=3)
ax.errorbar(sed_x_wavelength, sed_all_mean, yerr=sed_all_std,
            label='all', color='black', fmt='o', capsize=3)
ax.set_xlabel('Wavelength (10^-6 m)')
ax.set_ylabel('SED')
ax.legend()
ax_top = plt.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(sed_x_wavelength)
ax_top.set_xticklabels(sed_x_labels)
ax_top.xaxis.set_ticks_position('top')
ax_top.xaxis.set_label_position('top')
plt.setp(ax_top.get_xticklabels(), rotation=30,
         ha="left", rotation_mode="anchor")
plt.tight_layout()
plt.savefig(os.path.join(out_path, 'post_galaxy_sed_wspace.png'))
plt.close()

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
ax.errorbar(sed_x_wavelength, sed_galaxy_1_mean, yerr=sed_galaxy_1_std,
            label='galaxy 1', color='orange', fmt='o', capsize=3)
ax.errorbar(sed_x_wavelength, sed_galaxy_2_mean, yerr=sed_galaxy_2_std,
            label='galaxy 2', color='green', fmt='o', capsize=3)
ax.errorbar(sed_x_wavelength, sed_all_mean, yerr=sed_all_std,
            label='all', color='black', fmt='o', capsize=3)
ax.set_xlabel('Wavelength (10^-6 m)')
ax.set_ylabel('SED')
ax.legend()
ax_top = plt.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(sed_x_wavelength)
ax_top.set_xticklabels(sed_x_labels)
ax_top.xaxis.set_ticks_position('top')
ax_top.xaxis.set_label_position('top')
plt.setp(ax_top.get_xticklabels(), rotation=30,
         ha="left", rotation_mode="anchor")
plt.tight_layout()
plt.savefig(os.path.join(out_path, 'post_galaxy_sed.png'))
plt.close()


# Plotting samples
# norm = LogNorm(vmin=1e-4)
# plot_nd_array(full_samples, norm)

exit()

# Plotting prior
key = random.PRNGKey(42)
n_samples = 5
fm_priors = np.array([full_m(full_m.init(key+ii))
                      for ii in range(n_samples)])

norm = LogNorm(vmin=1e-4)
plot_nd_array(fm_priors, norm)

# Plot comparison

# f444w_smean, f444w_sstd, _, f444w_m, _ = load_reconstruction(f444w_path)
f1280w_smean, f1280w_sstd, _, f1280w_m, _ = load_reconstruction(f1280w_path)
f1500w_smean, f1500w_sstd, _, f1500w_m, _ = load_reconstruction(f1500w_path)
f1800w_smean, f1800w_sstd, _, f1800w_m, _ = load_reconstruction(f1800w_path)
f2100w_smean, f2100w_sstd, _, f2100w_m, _ = load_reconstruction(f2100w_path)
indiv_mean = dict(
    F2100W=f2100w_smean, F1800W=f1800w_smean, F1500W=f1500w_smean,
    F1280W=f1280w_smean)
indiv_std = dict(
    F2100W=f2100w_sstd, F1800W=f1800w_sstd, F1500W=f1500w_sstd,
    F1280W=f1280w_sstd)

norm_field = LogNorm(vmin=1e-2, vmax=0.6)
norm_std = Normalize(vmax=1.0)  # LogNorm(vmin=1e-3, vmax=1.0)
cmap_std = 'cubehelix'

fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
ims = np.zeros_like(axes)
for ii, fm, fs, fname in zip(range(4), full_smean, full_sstd, full_fp.keys_and_index.keys()):
    assert ii == full_fp.keys_and_index[fname]
    istd = indiv_std[fname][0]
    imean = indiv_mean[fname][0]
    axes[0, ii].set_title(fname)

    ims[0, ii] = axes[0, ii].imshow(
        fs/fm, origin='lower', extent=source_grid.extent(), norm=norm_std, cmap=cmap_std)
    ims[3, ii] = axes[3, ii].imshow(
        istd/imean, origin='lower', extent=source_grid.extent(), norm=norm_std, cmap=cmap_std)

    ims[1, ii] = axes[1, ii].imshow(
        fm, origin='lower', extent=source_grid.extent(), norm=norm_field)
    ims[2, ii] = axes[2, ii].imshow(
        imean, origin='lower', extent=source_grid.extent(), norm=norm_field)
for im, ax in zip(ims.flatten(), axes.flatten()):
    plt.colorbar(im, ax=ax)
plt.show()
