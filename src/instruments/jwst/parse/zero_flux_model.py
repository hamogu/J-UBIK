from .parametric_model.parametric_prior import (
    DefaultPriorConfig, UniformPriorConfig, DeltaPriorConfig,
    transform_setting_to_prior_config)

from dataclasses import dataclass
from typing import Union, Optional

DEFAULT_KEY = 'default'


@dataclass
class ZeroFluxPriorConfigs:
    default: Union[DefaultPriorConfig, UniformPriorConfig, DeltaPriorConfig]
    filters: dict[
        str, Union[DefaultPriorConfig, UniformPriorConfig, DeltaPriorConfig]
    ]


def yaml_to_zero_flux_prior_config(
    zero_flux_config: Optional[dict[str]]
):
    if zero_flux_config is None:
        return None

    default = transform_setting_to_prior_config(zero_flux_config[DEFAULT_KEY])

    filters = {}
    for filter_name, filter_prior in zero_flux_config.items():
        filter_name = filter_name.lower()
        if filter_name == DEFAULT_KEY:
            continue

        filters[filter_name] = transform_setting_to_prior_config(filter_prior)

    return ZeroFluxPriorConfigs(default=default, filters=filters)
