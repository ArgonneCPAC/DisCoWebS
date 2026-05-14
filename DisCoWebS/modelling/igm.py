import h5py
from astropy.table import Table
from jax import jit as jjit, numpy as jnp
from DisCoWebS import config
import os

IGM_ATTENUATION_BASENAME = "igm_attenuation.h5"


def load_igm_attenuation_table(drn=None, bn=IGM_ATTENUATION_BASENAME):
    """
    Load the IGM attenuation interpolation table from .h5 file.
    This table is used to apply IGM attenuation to the model photometry.
    """
    if drn is None:
        try:
            drn = os.environ["IGM_ATT_DRN"]
        except KeyError:
            msg = "Set environment variable IGM_ATT_DRN or pass drn argument"
            raise KeyError(msg)

    file = os.path.join(drn, bn)

    with h5py.File(file, "r") as f:
        table = {}
        for col in config.cosmos_web_filters_to_use:
            table[col + "/redshift"] = f[col + "/redshift"][:]
            table[col + "/igm_attenuation_inoue"] = f[col + "/igm_attenuation_inoue"][:]

    return table


@jjit
def igm_attenuation(phot_info_obs_mags, phot_info_z):
    phot_info_obs_mags_wIGM = phot_info_obs_mags.copy()

    for j, col in enumerate(config.cosmos_web_filters_to_use):
        igm_vals = jnp.interp(
            phot_info_z,
            config.igm_attenuation_table[col + "/redshift"],
            config.igm_attenuation_table[col + "/igm_attenuation_inoue"],
        )
        phot_info_obs_mags_wIGM = phot_info_obs_mags_wIGM.at[:, j].set(
            phot_info_obs_mags[:, j] + igm_vals
        )

    return phot_info_obs_mags_wIGM
