from abc import ABC
from typing import Union, Optional
from dataclasses import dataclass


class PriorConfig(ABC):
    distribution: str
    transformation: Optional[str]


@dataclass
class DefaultPriorConfig(PriorConfig):
    distribution: str
    mean: float
    sigma: float
    transformation: Optional[str] = None


@dataclass
class UniformPriorConfig(PriorConfig):
    distribution: str
    min: float
    max: float
    transformation: Optional[str] = None


@dataclass
class DeltaPriorConfig(PriorConfig):
    distribution: str
    mean: float
    transformation: Optional[str] = None


DISTRIBUTION_KEY = 'distribution'


def transform_setting_to_prior_config(parameters: Union[dict, tuple]):
    """Transforms the prior setting into a dictionary."""
    if isinstance(parameters, dict):
        return _set_prior_dict(parameters)

    elif isinstance(parameters, tuple) or isinstance(parameters, list):
        return _set_prior_tuple_or_list(parameters)

    raise NotImplementedError


def _set_prior_dict(parameters: dict[str, Union[str, float]]):
    '''Transforms tuple into a PriorConfig'''

    distribution = parameters[DISTRIBUTION_KEY]
    distribution = distribution.lower() if type(
        distribution) == str else distribution

    match distribution:
        case 'uniform':
            return UniformPriorConfig(**parameters)

        case 'delta' | None:
            return DeltaPriorConfig(**parameters)

        case _:
            return DefaultPriorConfig(**parameters)


def _set_prior_tuple_or_list(parameters: tuple[str, float, float]):
    '''Transforms tuple or list into a PriorConfig'''

    distribution = parameters[0]
    distribution = distribution.lower() if type(
        distribution) == str else distribution

    match distribution:
        case 'uniform':
            return UniformPriorConfig(
                distribution=distribution,
                min=parameters[1],
                max=parameters[2])

        case 'delta' | None:
            return DeltaPriorConfig(
                distribution=distribution,
                mean=parameters[1])

        case _:
            return DefaultPriorConfig(
                distribution=distribution,
                mean=parameters[1],
                sigma=parameters[2])
