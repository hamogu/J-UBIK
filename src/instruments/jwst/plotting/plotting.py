from .jwst_plotting import (
    get_alpha_nonpar,
    build_plot_lens_system,
    build_color_components_plotting,
    build_plot_sky_residuals,
    build_plot_source,
)
from ..filter_projector import FilterProjector

import nifty8.re as jft

from jax import random
from matplotlib.colors import LogNorm


def get_plot(
    results_directory: str,
    lens_system,
    filter_projector: FilterProjector,
    data_dict: dict,
    sky_model: jft.Model,
    sky_model_with_keys: jft.Model,
    cfg: dict,
    parametric_flag: bool
):
    ll_alpha, ll_nonpar, sl_alpha, sl_nonpar = get_alpha_nonpar(lens_system)

    plot_lens = build_plot_lens_system(
        results_directory,
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
        results_directory=results_directory,
        filter_projector=filter_projector,
        data_dict=data_dict,
        sky_model_with_key=sky_model_with_keys,
        small_sky_model=sky_model,
        plotting_config=dict(
            norm=LogNorm,
            data_config=dict(norm=LogNorm),
            display_pointing=False,
            xmax_residuals=cfg.get('max_residuals', 4),
        ),
    )
    plot_color = build_color_components_plotting(
        lens_system.source_plane_model.light_model.nonparametric(), results_directory, substring='source')

    plot_source = build_plot_source(
        results_directory,
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

    return plot_source, plot_residual, plot_color, plot_lens


def plot_prior(
    cfg: dict,
    likelihood: jft.Likelihood,
    filter_projector: FilterProjector,
    sky_model: jft.Model,
    sky_model_with_keys: jft.Model,
    plot_source: callable,
    plot_lens: callable,
    plot_color: callable,
    data_dict: dict,
    parametric_flag: bool,
):
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
        results_directory='',
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
        plot_lens(position, None, parametric=parametric_flag)
