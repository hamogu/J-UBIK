
from os.path import join, splitext, exists
import numpy as np
from jax import random, linear_transpose
from typing import NamedTuple


import nifty8.re as jft

from .erosita_observation import ErositaObservation
from .messages import log_file_exists
from .sky_models import SkyModel
from .utils import (get_config, create_output_directory, save_config_copy, save_dict_to_pickle,
                    load_vector_from_pickle)
from .plot import plot_result

class Domain(NamedTuple):
    """Mimicking NIFTy Domain.

    Parameters
    ----------
    shape: tuple
    distances: tuple
    """

    shape: tuple
    distances: tuple


def create_mock_data(tel_info, file_info, grid_info, prior_info, plot_info,
                             seed, response_dict):
    """ Generates and saves eROSITA mock data to pickle file.

    Parameters
    ----------
    tel_info : dict
        Dictionary of telescope information
    file_info : dict
        Dictionary of file paths
    grid_info : dict
        Dictionary with grid information
    grid_info : dict
        Dictionary with prior information
    plot_info: dict
        Dictionary with plotting information
    seed: int
        Random seed for mock position generataion
    response_dict : dict
        Dictionary of all available response functionalities i.e. response, mask, psf
    Returns
    -------
    masked_mock_data: jft.Vector
        Dictionary of masked mock data arrays
    """
    key = random.PRNGKey(seed)

    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    energy_range = np.array(e_max) - np.array(e_min)

    key, subkey = random.split(key)
    sky_model = SkyModel()
    sky = sky_model.create_sky_model(sdim=grid_info['sdim'], edim=grid_info['edim'],
                                     s_padding_ratio=grid_info['s_padding_ratio'],
                                     e_padding_ratio=grid_info['e_padding_ratio'],
                                     fov=tel_info['fov'],
                                     e_min=grid_info['energy_bin']['e_min'],
                                     e_max=grid_info['energy_bin']['e_max'],
                                     e_ref=grid_info['energy_bin']['e_ref'],
                                     priors=prior_info)

    jft.random_like(subkey, sky.domain)
    # Generate response func
    sky_comps = sky_model.sky_model_to_dict()
    key, subkey = random.split(key)
    output_path = create_output_directory(file_info['res_dir'])
    mock_sky_position = jft.Vector(jft.random_like(subkey, sky.domain))
    masked_mock_data = response_dict['R'](sky(mock_sky_position))
    subkeys = random.split(subkey, len(masked_mock_data.tree))
    masked_mock_data = jft.Vector({
        tm: random.poisson(subkeys[i], data).astype(int)
        for i, (tm, data) in enumerate(masked_mock_data.tree.items())
    })
    save_dict_to_pickle(masked_mock_data.tree, join(output_path, file_info['data_dict']))
    save_dict_to_pickle(mock_sky_position.tree, join(output_path, file_info['pos_dict']))
    if plot_info['enabled']:
        jft.logger.info('Plotting mock data and mock sky.')
        plottable_vector = jft.Vector({key: val.astype(float) for key, val
                                        in masked_mock_data.tree.items()})
        mask_adj = linear_transpose(response_dict['mask'],
                                    np.zeros((len(tel_info['tm_ids']),) + sky.target.shape))
        mask_adj_func = lambda x: mask_adj(x)[0]
        plottable_data_array = np.stack(mask_adj_func(plottable_vector), axis=0)
        from .mf_plot import plot_rgb
        for tm_id in range(plottable_data_array.shape[0]):
            plot_rgb(plottable_data_array[tm_id],
                    name=join(output_path, f'mock_data_tm_rgb_log{tm_id+1}'), log=True)
            plot_rgb(plottable_data_array[tm_id],
                    name=join(output_path, f'mock_data_tm_rgb_{tm_id+1}'), log=False,
                    sat_min=(np.min(plottable_data_array[0], axis=(1, 2))).tolist(),
                    sat_max=(0.1*np.max(plottable_data_array[0], axis=(1, 2))).tolist())
            plot_result(plottable_data_array[tm_id], logscale=True,
                    output_file=join(output_path, f'mock_data_tm{tm_id+1}.png'))
        for key, sky_comp in sky_comps.items():
            plot_rgb(sky_comp(mock_sky_position),
                     name=join(output_path, f'mock_rgb_log_{key}'), log=True)
            plot_rgb(sky_comp(mock_sky_position),
                    name=join(output_path, f'mock_rgb_{key}'), log=False,
                     sat_min=(np.min(sky_comp(mock_sky_position), axis=(1, 2))).tolist(),
                     sat_max=(0.1 * np.max(sky_comp(mock_sky_position), axis=(1, 2))).tolist())
            plot_result(sky_comp(mock_sky_position), logscale=True,
                    output_file=join(output_path, f'mock_{key}.png'))
        if hasattr(sky_model, 'alpha_cf'):
            diffuse_alpha = sky_model.alpha_cf
            plot_result(diffuse_alpha(mock_sky_position), logscale=False,
                    output_file=join(output_path, f'mock_diffuse_alpha.png'))
    return masked_mock_data


# Data creation wrapper
def create_data_from_config(config_path, response_dct):
    """ Wrapper function to create masked data either from
    actual eROSITA observations or from generated mock data, as specified
    in the config given at config path. In any case the data is saved to the
    same pickle file.

    Parameters
    ----------
    config_path : str
        Path to inference config file

    """
    cfg = get_config(config_path)

    tel_info = cfg["telescope"]
    file_info = cfg["files"]
    grid_info = cfg['grid']
    plot_info = cfg['plotting']
    data_path = join(file_info['res_dir'], file_info['data_dict'])
    if not exists(data_path):
        if bool(file_info.get("mock_gen_config")):
            jft.logger.info(f'Generating new mock data in {file_info["res_dir"]}...')
            mock_prior_info = get_config(file_info["mock_gen_config"])
            _ = create_mock_erosita_data(tel_info, file_info, grid_info, mock_prior_info,
                                         plot_info, cfg['seed'], response_dct)
            save_config_copy(file_info['mock_gen_config'], output_dir=file_info['res_dir'])
        else:
            jft.logger.info(f'Generating masked eROSITA data in {file_info["res_dir"]}...')
            mask_erosita_data_from_disk(file_info, tel_info, grid_info, response_dct['mask'])
    else:
        jft.logger.info(f'Data in {file_info["res_dir"]} already exists. No data generation.')


# Data loading wrapper
def load_masked_data_from_config(config_path):
    """ Wrapper function load masked eROSITA data from config path
        from generated pickle-files.

    Parameters
    ----------
    config_path : str
        Path to inference config file

    Returns
    ----------
    masked data: jft.Vector
        Vector of masked eROSITA (mock) data for each TM
    """
    cfg = get_config(config_path)
    file_info = cfg['files']
    data_path = join(file_info['res_dir'], file_info['data_dict'])
    if exists(data_path):
        jft.logger.info('...Loading data from file')
        masked_data = jft.Vector(load_vector_from_pickle(data_path))
    else:
        raise ValueError('Data path does not exist.')
    return masked_data


def load_mock_position_from_config(config_path):
    """ Wrapper function to load the mock sky position for the eROSITA mock data config path
        from pickle-file.

    Parameters
    ----------
    config_path : str
        Path to inference config file

    Returns
    ----------
    mock_pos : jft.Vector
        Vector of latent parameters for the mock sky position

    """
    cfg = get_config(config_path)
    file_info = cfg['files']
    pos_path = join(file_info['res_dir'], file_info['pos_dict'])
    if exists(pos_path):
        jft.logger.info('...Loading mock position')
        mock_pos = load_vector_from_pickle(pos_path)
    else:
        raise ValueError('Mock position path does not exist.')
    return mock_pos




