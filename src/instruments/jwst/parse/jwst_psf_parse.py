from ..jwst_data import JwstData

from astropy.coordinates import SkyCoord
from dataclasses import dataclass


WEBBPSF_PATH_KEY = 'webbpsf_path'
PSF_LIBRARY_PATH_KEY = 'psf_library'
PSF_ARCSEC_KEY = 'psf_arcsec_extension'
NORMALIZE_KEY = 'normalize'
NORMALIZE_DEFAULT = 'last'


@dataclass
class PsfKernelConfigParameters:
    '''The PsfKernelConfigParameters is a data model for holding metadata for
    the evaluation of the psf kernel.

    webbpsf_path : str
        The path to the directory containing the `webbpsf` data files.
    psf_library_path : str
        The directory where the computed PSF files are stored.
        The PSF kernel will be saved here if it is not already present.
    psf_arcsec : float
        The size of the PSF evaluation in arcsec. If not provided, `psf_pixels`
        must be specified.
    normalize : str
        The normalization method for the PSF. Default is 'last',
        but other methods may be supported by `webbpsf`.
    '''
    webbpsf_path: str
    psf_library_path: str
    psf_arcsec: float
    normalize: str


def yaml_to_psf_kernel_config_parameters(
    psf_config: dict,
):
    '''Read the PsfKernelConfigParameters from the yaml config.

    psf_config: dict
        The dictionary holding:
            - webbpsf_path
            - psf_library_path
            - psf_arcsec_extension
            - normalize (Optional)
    '''

    return PsfKernelConfigParameters(
        webbpsf_path=psf_config[WEBBPSF_PATH_KEY],
        psf_library_path=psf_config[PSF_LIBRARY_PATH_KEY],
        psf_arcsec=psf_config[PSF_ARCSEC_KEY],
        normalize=psf_config.get(NORMALIZE_KEY, 'last'),
    )


@dataclass
class PsfKernelModel:
    '''The PsfKernelModel is a data model for the evaluation of the psf kernel.

    Parameters
    ----------
    camera : str
        The camera model for which to compute the PSF.
        Options are 'nircam' and 'miri'.
        This value is converted to lowercase before processing.
    filter : str
        The filter for which to compute the PSF.
    center_pixel : tuple of float
        The (x, y) coordinates of the center pixel for the PSF calculation.
    subsample : int
        The subsample factor for the PSF computation.
    config_parameters: PsfKernelConfigParameters
        Metadata for the evaluation of the psf kernel:
            - webbpsf_path : str
            - psf_library_path : str
            - psf_arcsec : float
            - normalize : str
    '''
    camera: str
    filter: str
    pointing_center: tuple[float, float]
    subsample: int
    config_parameters: PsfKernelConfigParameters

    @classmethod
    def from_jwst_pointing_subsample_and_config(
        cls,
        jwst_data: JwstData,
        pointing_center: SkyCoord,
        subsample: int,
        config_parameters: PsfKernelConfigParameters
    ):
        '''Initialization for the `PsfKernelModel`.

        Parameters
        ----------
        jwst_data: JwstData
            jwst data with camera and filter
        pointing_center: SkyCoord
            The center of the observation for which the psf will be evaluted.
            The psf kernel is assumed to be static across the field.
        subsample: int
            The subsample factor for the psf kernel.
        '''

        pointing_center = jwst_data.wcs.index_from_wl(pointing_center)[0]

        return PsfKernelModel(
            camera=jwst_data.camera,
            filter=jwst_data.filter,
            pointing_center=pointing_center,
            subsample=subsample,
            config_parameters=config_parameters,
        )
