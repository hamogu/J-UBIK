from sys import exit
import nifty8.re as jft

import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from jwst_handling.interpolation_models import (
    build_sparse_interpolation,
    build_sparse_interpolation_model,
)
import jubik0 as ju

import jax.numpy as jnp
from jax import config, random
config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')


def downscale_sum(high_res_array, reduction_factor):
    """
    Sums the entries of a high-resolution array into a lower-resolution array
    by the given reduction factor.

    Parameters:
    - high_res_array: np.ndarray, the high-resolution array to be downscaled.
    - reduction_factor: int, the factor by which to reduce the resolution.

    Returns:
    - A lower-resolution array where each element is the sum of a block from the
      high-resolution array.
    """
    # Ensure the reduction factor is valid
    if high_res_array.shape[0] % reduction_factor != 0 or high_res_array.shape[1] % reduction_factor != 0:
        raise ValueError(
            "The reduction factor must evenly divide both dimensions of the high_res_array.")

    # Reshape and sum
    new_shape = (high_res_array.shape[0] // reduction_factor, reduction_factor,
                 high_res_array.shape[1] // reduction_factor, reduction_factor)
    return high_res_array.reshape(new_shape).sum(axis=(1, 3))


def create_data(key, shapes, mock_dist, model_setup, show=True):
    offset, fluctuations = model_setup

    cfm = jft.CorrelatedFieldMaker(prefix='mock')
    cfm.set_amplitude_total_offset(**offset)
    cfm.add_fluctuations(
        mock_shape, mock_dist, **fluctuations, non_parametric_kind='power')
    mock_diffuse = cfm.finalize()
    mock_sky = jnp.exp(mock_diffuse(
        jft.random_like(mock_key, mock_diffuse.domain)))

    comparison_sky = downscale_sum(mock_sky, mock_shape[0] // reco_shape[0])
    data = downscale_sum(mock_sky, mock_shape[0] // data_shape[0])
    mask = np.full(data_shape, True, dtype=bool)

    # comparison_sky = np.sum(
    #     [mock_sky[..., ii::down, ii::down] for ii in range(down)], axis=0)
    # down = mock_shape[0] // data_shape[0]
    # data = np.sum(
    #     [mock_sky[..., ii::down, ii::down] for ii in range(down)], axis=0)

    if show:
        fig, axes = plt.subplots(1, 3)
        ims = []
        ims.append(axes[0].imshow(mock_sky, origin='lower'))
        ims.append(axes[1].imshow(comparison_sky,
                   origin='lower'))
        ims.append(axes[2].imshow(data, origin='lower'))
        axes[0].set_title('Mock sky')
        axes[1].set_title('Comparison sky')
        axes[2].set_title('Data')
        for im, ax in zip(ims, axes):
            fig.colorbar(im, ax=ax, shrink=0.7)
        plt.show()

    return mock_sky, comparison_sky, data, mask


key = random.PRNGKey(42)
key, mock_key, rec_key = random.split(key, 3)

offset = dict(offset_mean=0.1, offset_std=[0.1, 0.05])
fluctuations = dict(fluctuations=[0.3, 0.03],
                    loglogavgslope=[-3., 1.],
                    flexibility=[0.8, 0.1],
                    asperity=[0.2, 0.1])

mock_shape, mock_dist = (1024, 1024), (0.5, 0.5)
reco_shape, reco_dist = (128, 128), (4.0, 4.0)
data_shape, data_dist = (64, 64), (8.0, 8.0)

mock_sky, comparison_sky, data, mask = create_data(
    mock_key, (mock_shape, reco_shape, data_shape), mock_dist,
    (offset, fluctuations), show=False
)

offset = dict(offset_mean=3.7, offset_std=[0.1, 0.05])
fluctuations = dict(fluctuations=[0.7, 0.03],
                    loglogavgslope=[-4.8, 1.],
                    flexibility=[0.8, 0.1],
                    asperity=[0.2, 0.1])
padding = 1.5
cfm = jft.CorrelatedFieldMaker(prefix='reco')
cfm.set_amplitude_total_offset(**offset)
cfm.add_fluctuations(
    [int(shp * padding) for shp in reco_shape],
    mock_dist, **fluctuations, non_parametric_kind='power')
reco_diffuse = cfm.finalize()
sky_model = jft.Model(
    lambda x: jnp.exp(reco_diffuse(x)[:reco_shape[0], :reco_shape[1]]),
    domain=reco_diffuse.domain)
sky_model_full = jft.Model(
    lambda x: jnp.exp(reco_diffuse(x)),
    domain=reco_diffuse.domain)

check = True
if check:
    key, check_key = random.split(key)
    m = sky_model(jft.random_like(rec_key, sky_model.domain))
    fig, axis = plt.subplots(1, 3)
    im0 = axis[0].imshow(comparison_sky, origin='lower')
    im1 = axis[1].imshow(m, origin='lower')
    im2 = axis[2].imshow(comparison_sky-m, origin='lower', cmap='RdBu_r')
    axis[0].set_title('sky')
    axis[1].set_title('model')
    axis[2].set_title('residual')
    for im, ax in zip([im0, im1, im2], axis):
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.show()


def get_sparse_model(mask):
    # Sparse Interpolation
    extent = [int(s * padding) for s in reco_shape]
    extent = [(e - s) // 2 for s, e in zip(reco_shape, extent)]
    x, y = [np.arange(-e, s+e) for e, s in zip(extent, reco_shape)]
    x, y = np.roll(x, -extent[0]), np.roll(y, -extent[1])  # to bottom left
    index_grid_reco = np.meshgrid(x, y, indexing='xy')

    # Index Edges
    factor = [int(dd/rd) for dd, rd in zip(data_dist, reco_dist)]
    pix_center = np.array(np.meshgrid(
        *[np.arange(ds)*f for ds, f in zip(data_shape, factor)]
    )) + np.array([1/f for f in factor])[..., None, None]
    e00 = pix_center - np.array([0.5*factor[0], 0.5*factor[1]])[:, None, None]
    e01 = pix_center - np.array([0.5*factor[0], -0.5*factor[1]])[:, None, None]
    e10 = pix_center - np.array([-0.5*factor[0], 0.5*factor[1]])[:, None, None]
    e11 = pix_center - \
        np.array([-0.5*factor[0], -0.5*factor[1]])[:, None, None]
    index_edges = np.array([e00, e01, e11, e10])

    sparse_matrix = build_sparse_interpolation(
        index_grid_reco, index_edges, mask)
    sparse_model = build_sparse_interpolation_model(
        sparse_matrix, sky_model_full)
    return sparse_model


std = 0.05*data.mean()
d = data + np.random.normal(scale=std, size=data.shape)
# plt.imshow(d, origin='lower')
# plt.show()

model = get_sparse_model(mask)

like = ju.library.likelihood.build_gaussian_likelihood(
    d.reshape(-1), float(std))
like = like.amend(model, domain=model.domain)


def build_plot(plot_data, plot_sky, mask, data_model, sky_model, res_dir):
    from charm_lensing.analysis_tools import source_distortion_ratio
    from scipy.stats import wasserstein_distance
    from charm_lensing.plotting import display_text

    def plot(s, x):
        from os.path import join
        from os import makedirs
        out_dir = join(res_dir, 'residuals')
        makedirs(out_dir, exist_ok=True)

        mod = np.zeros_like(plot_data)
        mod[mask] = jft.mean([data_model(si) for si in s])
        sky = jft.mean([sky_model(si) for si in s])

        vals = dict(
            sdr=source_distortion_ratio(plot_sky, sky),
            # wd=wasserstein_distance(plot_sky, sky)
        )

        fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=300)
        ims = []
        axes[0, 0].set_title('Data')
        ims.append(axes[0, 0].imshow(plot_data, origin='lower'))
        axes[0, 1].set_title('Data model')
        ims.append(axes[0, 1].imshow(mod, origin='lower'))
        axes[0, 2].set_title('Data residual')
        ims.append(axes[0, 2].imshow((plot_data - mod)/std, origin='lower',
                                     vmin=-3, vmax=3, cmap='RdBu_r'))
        axes[1, 0].set_title('Sky')
        ims.append(axes[1, 0].imshow(plot_sky, origin='lower'))
        axes[1, 1].set_title('Sky model')
        ims.append(axes[1, 1].imshow(sky, origin='lower'))
        axes[1, 2].set_title('Sky residual')
        ims.append(axes[1, 2].imshow((plot_sky - sky)/plot_sky, origin='lower',
                                     vmin=-1, vmax=1, cmap='RdBu_r'))
        for ii, (k, v) in enumerate(vals.items()):
            display_text(axes[1, 2], dict(
                s=f'{k}: {v:.3f}', color='black'), y_offset_ticker=ii)

        for ax, im in zip(axes.flatten(), ims):
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        # plt.show()
        fig.savefig(join(out_dir, f'{x.nit:02d}.png'), dpi=300)
        plt.close()

    return plot


res_dir = 'results/mock_interpolation/sparse'
plot = build_plot(d, comparison_sky, mask,
                  model, sky_model, res_dir)

pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, like.domain))

cfg = ju.get_config('./JWST_config.yaml')
minimization_config = cfg['minimization']
kl_solver_kwargs = minimization_config.pop('kl_kwargs')
minimization_config['n_total_iterations'] = 15

samples, state = jft.optimize_kl(
    like,
    pos_init,
    key=key,
    kl_kwargs=kl_solver_kwargs,
    callback=plot,
    odir=res_dir,
    **minimization_config)
