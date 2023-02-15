import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import nifty8 as ift

from src.library.utils import save_rgb_image_to_fits


def get_uncertainty_weighted_measure(sl, op=None,
                                     reference=None,
                                     output_dir_base=None,
                                     title=''):
    mpi_master = ift.utilities.get_MPI_params()[3]
    mean, var = sl.sample_stat(op)
    if reference is None:
        wgt_res = mean / var.sqrt()
    else:
        wgt_res = (mean-reference).abs() / var.sqrt()
    if mpi_master and output_dir_base is not None:
        with open(f'{output_dir_base}.pkl', 'wb') as file:
            pickle.dump(wgt_res, file)
        save_rgb_image_to_fits(wgt_res, output_dir_base,
                               overwrite=True, MPI_master=mpi_master)
        p = ift.Plot()
        p.add(wgt_res, title=f"Uncertainty weighted {title}", norm=LogNorm())
        p.output(name=f'{output_dir_base}.png')
    return wgt_res


def signal_space_uwr_from_file(sl_path_base,
                               ground_truth_path,
                               sky_op,
                               output_dir_base=None,
                               title='signal space UWR'):
    sl = ift.ResidualSampleList.load(sl_path_base)
    with open(ground_truth_path, "rb") as f:
         gt = pickle.load(f)
    wgt_res = get_uncertainty_weighted_measure(sl, sky_op, gt, output_dir_base,
                                               title=title)
    return wgt_res


def data_space_uwr_from_file(sl_path_base,
                             data_path,
                             sky_op,
                             response_op,
                             output_dir_base=None,
                             title='data space UWR'):
    sl = ift.ResidualSampleList.load(sl_path_base)
    with open(data_path, "rb") as f:
         d = pickle.load(f)
    wgt_res = get_uncertainty_weighted_measure(sl, response_op @ sky_op, d, output_dir_base,
                                               title=title)
    return wgt_res


def signal_space_uwm_from_file(sl_path_base,
                               sky_op,
                               output_dir_base=None,
                               title='signal space UWM'):
    sl = ift.ResidualSampleList.load(sl_path_base)
    wgt_mean = get_uncertainty_weighted_measure(sl, sky_op, output_dir_base=output_dir_base,
                                                title=title)
    return wgt_mean


def weighted_residual_distribution(sl_path_base,
                                   data_path,
                                   sky_op,
                                   response_op,
                                   bins=20,
                                   output_dir_base=None,
                                   title='Weighted data residuals'):
    sl = ift.ResidualSampleList.load(sl_path_base)
    with open(data_path, "rb") as f:
        d = pickle.load(f)
    wgt_res = get_uncertainty_weighted_measure(sl, response_op @ sky_op, d, None,
                                               title=title)
    res = wgt_res.val.reshape(-1)
    plt.histogram(res, bins)
    plt.savefig(fname=title)
    return
