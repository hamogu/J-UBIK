# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

from .parametric_model import build_parametric_prior_from_prior_config
from .parse.zero_flux_model import ZeroFluxPriorConfigs
from .parse.parametric_model.parametric_prior import PriorConfig

import nifty8.re as jft

from typing import Union, Optional

ZERO_FLUX_KEY = 'zero_flux'
DEFAULT_KEY = 'default'
SHAPE = (1,)


def build_zero_flux_model(
    prefix: str,
    prior_config: Optional[PriorConfig],
) -> jft.Model:
    """
    Build a zero flux model based on the provided likelihood configuration.

    If no specific configuration for the zero-flux model is found in
    `likelihood_config`, a default model returning zero is created.
    Otherwise, it builds a parametric model based on the provided configuration.

    Parameters
    ----------
    prefix : str
        A string prefix used to identify and name the parameters associated with
         the zero-flux model.
    likelihood_config : dict
        A configuration dictionary containing model details.
        The zero-flux model configuration is expected to be under the key
        specified by `ZERO_FLUX_KEY`.

    Returns
    -------
    jft.Model
        A model representing the zero flux configuration.
        If no configuration is provided, the model returns zero;
        otherwise, it uses a parametric prior.
    """
    if prior_config is None:
        return None

    prefix = '_'.join([prefix, ZERO_FLUX_KEY])

    prior = build_parametric_prior_from_prior_config(
        prefix, prior_config, SHAPE)
    return jft.Model(prior, domain={prefix: jft.ShapeWithDtype(SHAPE)})


def get_filter_or_default_prior(
    filter_name: str,
    zero_flux_prior_config: Optional[ZeroFluxPriorConfigs] = None,
) -> Union[DefaultPriorConfig, UniformPriorConfig, DeltaPriorConfig]:
    '''Returns the PriorConfig for the `filter_name` or the default.'''

    if zero_flux_prior_config is None:
        return None

    if filter_name in zero_flux_prior_config.filters:
        return zero_flux_prior_config.filters[filter_name]
    return zero_flux_prior_config.default
