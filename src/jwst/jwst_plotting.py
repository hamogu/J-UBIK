import nifty8.re as jft

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np

from typing import Optional


def find_closest_factors(number):
    """
    Finds two integers whose multiplication is larger or equal to the input number,
    with the two output numbers being as close together as possible.

    Args:
        number: The input integer number.

    Returns:
        A tuple containing two integers (x, y) such that x * y >= number and the
        difference between x and y is minimized. If no such factors exist, returns
        None.
    """

    # Start with the square root of the number.
    ii = int(np.ceil(number**0.5))

    jj, kk = ii, ii

    jminus = kminus = 0
    while ((jj-jminus)*(kk-kminus) >= number):
        if kminus == jminus:
            jminus += 1
        else:
            kminus += 1

    if jminus == kminus:
        return jj-jminus, kk-kminus+1
    return jj-jminus+1, kk-kminus


def build_plot(
    data_dict: dict,
    sky_model_with_key: jft.Model,
    sky_model: jft.Model,
    small_sky_model: jft.Model,
    results_directory: str,
    plotting_config: dict,
    plaw: Optional[jft.Model] = None,
):
    from jubik0.jwst.mock_data.mock_evaluation import redchi2
    from jubik0.jwst.mock_data.mock_plotting import display_text
    from os.path import join
    from os import makedirs

    residual_dir = join(results_directory, 'residuals')
    sky_dir = join(results_directory, 'sky')
    makedirs(residual_dir, exist_ok=True)
    makedirs(sky_dir, exist_ok=True)

    if plaw is not None:
        plaw_dir = join(results_directory, 'plaw')
        makedirs(plaw_dir, exist_ok=True)

    norm = plotting_config.get('norm', Normalize)
    sky_extent = plotting_config.get('sky_extent', None)

    def sky_plot_residuals(samples: jft.Samples, x: jft.OptimizeVIState):
        print(f"Results: {results_directory}")

        ylen = len(data_dict)
        fig, axes = plt.subplots(ylen, 3, figsize=(9, 3*ylen), dpi=300)
        ims = []
        for ii, (dkey, data) in enumerate(data_dict.items()):

            dm = data['data_model']
            dd = data['data']
            std = data['std']
            mask = data['mask']
            cm = data['correction_model']

            model_data = []
            for si in samples:
                tmp = np.zeros_like(dd)
                val = sky_model_with_key(si)
                if cm is not None:
                    val = val | cm(si)
                tmp[mask] = dm(val)
                model_data.append(tmp)

            if cm is not None:
                corr, cors = jft.mean_and_std(
                    [cm.prior_model(s) for s in samples])
                corr, cors = (next(iter(corr.values())).reshape(2),
                              next(iter(cors.values())).reshape(2))
            else:
                corr, cors = (0, 0), (0, 0)

            model_mean = jft.mean(model_data)
            redchi_mean, redchi2_std = jft.mean_and_std(
                [redchi2(dd[mask], m[mask], std[mask], dd[mask].size) for m in model_data])

            axes[ii, 0].set_title(f'Data {dkey}')
            ims.append(axes[ii, 0].imshow(dd, origin='lower', norm=norm()))
            axes[ii, 1].set_title(
                f'Data model ({corr[0]:.1e}+-{cors[0]:.1e}, {corr[1]:.1e}+-{cors[1]:.1e})')
            ims.append(axes[ii, 1].imshow(
                model_mean, origin='lower', norm=norm()))
            axes[ii, 2].set_title('Data - Data model')
            ims.append(axes[ii, 2].imshow((dd - model_mean)/std,
                       origin='lower', vmin=-3, vmax=3, cmap='RdBu_r'))

            chi = '\n'.join((
                f'redChi2: {redchi_mean:.2f} +/- {redchi2_std:.2f}',
            ))

            display_text(axes[ii, 2], chi)

        for ax, im in zip(axes.flatten(), ims):
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(join(residual_dir, f'{x.nit:02d}.png'), dpi=300)
        plt.close()

    def plot_plaw(samples: jft.Samples, x: jft.OptimizeVIState):
        m_plaw, s_plaw = jft.mean_and_std([plaw(si) for si in samples])
        m_sky, s_sky = jft.mean_and_std(
            [small_sky_model(si) for si in samples])

        ylen = len(data_dict)
        fig, axes = plt.subplots(ylen, 3, figsize=(9, 3*ylen), dpi=300)
        ims = []
        for ii, (dkey, data) in enumerate(data_dict.items()):
            dm = data['data_model']
            dd = data['data']
            std = data['std']
            mask = data['mask']
            index = data['index']
            cm = data['correction_model']

            model_data = []
            for si in samples:
                tmp = np.zeros_like(dd)
                val = sky_model_with_key(si)
                if cm is not None:
                    val = val | cm(si)
                tmp[mask] = dm(val)
                model_data.append(tmp)

            if cm is not None:
                corr, cors = jft.mean_and_std(
                    [cm.prior_model(s) for s in samples])
                corr, cors = (next(iter(corr.values())).reshape(2),
                              next(iter(cors.values())).reshape(2))
            else:
                corr, cors = (0, 0), (0, 0)

            model_mean = jft.mean(model_data)

            print(index)
            axes[ii, 0].set_title('p-law model')
            ims.append(axes[ii, 0].imshow(m_plaw[index], origin='lower'))
            axes[ii, 1].set_title(
                f'Data model ({corr[0]:.1e}+-{cors[0]:.1e}, {corr[1]:.1e}+-{cors[1]:.1e})')
            ims.append(axes[ii, 1].imshow(
                m_sky[index], origin='lower', norm=norm()))
            axes[ii, 2].set_title('Data - Data model')
            ims.append(axes[ii, 2].imshow((dd - model_mean)/std,
                       origin='lower', vmin=-3, vmax=3, cmap='RdBu_r'))

        for ax, im in zip(axes.flatten(), ims):
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(join(plaw_dir, f'{x.nit:02d}.png'), dpi=300)
        plt.close()

    def plot_sky_with_samples(samples: jft.Samples, x: jft.OptimizeVIState):
        ylen, xlen = find_closest_factors(len(samples)+4)

        samps_big = [sky_model(si) for si in samples]
        mean, std = jft.mean_and_std(samps_big)
        mean_small, std_small = jft.mean_and_std(
            [small_sky_model(si) for si in samples])
        flds = [mean_small, std_small/mean_small, mean, std/mean] + samps_big

        for ii, filter_name in enumerate(sky_model_with_key.target.keys()):
            fig, axes = plt.subplots(
                ylen, xlen, figsize=(2*xlen, 1.5*ylen), dpi=300)
            for ax, fld in zip(axes.flatten(), flds):
                im = ax.imshow(
                    fld[ii], origin='lower', norm=norm(), extent=sky_extent)
                fig.colorbar(im, ax=ax, shrink=0.7)
            fig.tight_layout()
            fig.savefig(
                join(sky_dir, f'{x.nit:02d}_{filter_name}.png'), dpi=300)
            plt.close()

    def sky_plot(samples: jft.Samples, x: jft.OptimizeVIState):
        print(f'Plotting: {x.nit}')
        sky_plot_residuals(samples, x)
        plot_sky_with_samples(samples, x)
        if plaw is not None:
            plot_plaw(samples, x)

    return sky_plot


def plot_sky(sky, data_dict, norm=LogNorm):

    ylen = len(data_dict)
    fig, axes = plt.subplots(ylen, 3, figsize=(9, 3*ylen), dpi=300)
    ims = []
    for ii, (dkey, data) in enumerate(data_dict.items()):

        dm = data['data_model']
        dd = data['data']
        std = data['std']
        mask = data['mask']

        model_data = np.zeros_like(dd)
        model_data[mask] = dm(sky)

        axes[ii, 0].set_title(f'Data {dkey}')
        ims.append(axes[ii, 0].imshow(dd, origin='lower', norm=norm()))
        axes[ii, 1].set_title('Data model')
        ims.append(axes[ii, 1].imshow(
            model_data, origin='lower', norm=norm()))
        axes[ii, 2].set_title('Data - Data model')
        ims.append(axes[ii, 2].imshow((dd - model_data)/std,
                   origin='lower', vmin=-3, vmax=3, cmap='RdBu_r'))

    for ax, im in zip(axes.flatten(), ims):
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.show()
