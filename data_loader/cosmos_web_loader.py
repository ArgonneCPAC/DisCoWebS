"""Module implements the `load_cosmos_web` functions"""

from DisCoWebS import config
import os

import numpy as np
from functools import partial
import jax
from jax import numpy as jnp

try:
    from astropy.table import Table, hstack
    from astropy.io import fits

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


NANFILL = -999.0

COSMOS_web_photom_BASENAME = "COSMOSWeb_mastercatalog_v1_photom_primary.fits"
COSMOS_web_lephare_BASENAME = "COSMOSWeb_mastercatalog_v1_lephare.fits"

SKY_AREA = 0.433  # square degrees

__all__ = ("load_cosmos_web_without_MIRI",)


def load_cosmos_web_without_MIRI(
    drn=None,
    bn_photom=COSMOS_web_photom_BASENAME,
    bn_lephare=COSMOS_web_lephare_BASENAME,
    apply_cuts=True,
    app_mag_f444w_cut=27.0,
    z_min=0.1,
    z_max=7.0,
):
    """Load the COSMOS_web dataset from disk and calculate quality cuts

    Parameters
    ----------
    drn : string, optional
        Absolute path to directory containing .fits file storing COSMOS_web dataset
        Default value is os.environ['COSMOS_web_DRN'].

        For bash users, add the following line to your `.bash_profile` in order to
        configure the package to use your default dataset location:

        export COSMOS_web_DRN="/drn/storing/COSMOS_web"

    bn : string, optional
        Absolute path to directory containing .fits file storing COSMOS_web dataset
        Default value is COSMOS_web_BASENAME set at top of module

    apply_cuts : bool, optional
        If True, returned Table will have quality cuts imposed on the data
        Default is True

    app_mag_f444w_cut : float, optional
        Faintest apparent magnitude in the F444W band
        for a galaxy to be included in the returned catalog.

    Returns
    -------
    cat : astropy.table.Table
        Table of length ngals

    Notes
    -----
    Quality cuts include type=0 for the `galaxies` flag.
    warn_flag=0 for "no warning flag",
    mag_model_f444w < app_mag_f444w_cut, to define the faint end,
    flag_star_hsc=0 to remove objects in the HSC star mask area,
    and z_min <= zfinal <= z_max to remove objects outside the z of interest.

    The sky area with these quality cuts is approximately 0.433 square degrees,
    as computed by K. Mitra.

    """
    if not HAS_ASTROPY:
        raise ImportError("Must have astropy installed to use cosmos_web_loader.py")

    if drn is None:
        try:
            drn = os.environ["COSMOS_web_DRN"]
        except KeyError:
            msg = "Must set environment variable COSMOS_web_DRN or pass drn argument"
            raise KeyError(msg)

    fn_photom = os.path.join(drn, bn_photom)
    with fits.open(fn_photom) as hdu:
        cat_photom = Table(hdu[1].data)

    fn_lephare = os.path.join(drn, bn_lephare)
    with fits.open(fn_lephare) as hdu:
        cat_lephare = Table(hdu[1].data)

    cat = hstack([cat_photom, cat_lephare], join_type="exact")

    cat_photom_columns_to_keep = [
        "mag_model_cfht-u",
        "mag_model_hsc-g",
        "mag_model_hsc-r",
        "mag_model_hsc-i",
        "mag_model_hsc-z",
        "mag_model_hsc-y",
        "mag_model_hst-f814w",
        "mag_model_f115w",
        "mag_model_f150w",
        "mag_model_f277w",
        "mag_model_f444w",
        "mag_model_f770w",
    ]

    cat_lephare_columns_to_keep = [
        "zfinal",
        "zpdf_med",
        "zpdf_l68",
        "zpdf_u68",
        "mass_minchi2",
        "sfr_minchi2",
        "ssfr_minchi2",
        "mass_l68",
        "mass_med",
        "mass_u68",
        "sfr_l68",
        "sfr_med",
        "sfr_u68",
        "ssfr_l68",
        "ssfr_med",
        "ssfr_u68",
    ]

    cat.keep_columns(cat_photom_columns_to_keep + cat_lephare_columns_to_keep)

    if apply_cuts:
        condition_clean = np.logical_and.reduce(
            (
                cat_lephare["type"] == 0,  # Select only galaxies
                cat_photom["warn_flag"] == 0,  # No warning flag
                np.abs(cat_photom["mag_model_f444w"])
                < app_mag_f444w_cut,  # Remove very faint objects
                cat_photom["flag_star_hsc"]
                == 0,  # Remove objects in HSC star mask area
                cat_lephare["zfinal"] >= z_min,  # Remove z < z_min
                cat_lephare["zfinal"] <= z_max,  # Remove z > z_max
            )
        )

        return cat[condition_clean]

    else:
        return cat


def impose_cosmos_web_mag_cut(cosmos_web_subset, cosmos_web_mag_colnames):
    final_mask = np.ones(len(cosmos_web_subset), dtype=bool)

    filter_thresholds = {
        "mag_model_cfht-u": 27.0,
        "mag_model_hsc-g": 27.0,
        "mag_model_hsc-r": 27.0,
        "mag_model_hsc-i": 27.0,
        "mag_model_hsc-z": 26.5,
        "mag_model_hsc-y": 26.0,
        "mag_model_hst-f814w": 27.0,
        "mag_model_f115w": 27.0,
        "mag_model_f150w": 27.0,
        "mag_model_f277w": 27.0,
        "mag_model_f444w": 27.0,
        "mag_model_f770w": 27.0,
    }

    for filter_name, mmax in filter_thresholds.items():
        if filter_name in cosmos_web_mag_colnames:
            i = cosmos_web_mag_colnames.index(filter_name)
            final_mask &= cosmos_web_subset[cosmos_web_mag_colnames[i]] < mmax

    return cosmos_web_subset[final_mask]


def assign_dropout_values_CW_data(
    cosmos_web_4d_data,
    cosmos_web_mag_colnames,
    mi,
    mi2,
    mi1,
    cosmos_web_subset,
):

    filter_thresholds = {
        "mag_model_cfht-u": 27.0,
        "mag_model_hsc-g": 27.0,
        "mag_model_hsc-r": 27.0,
        "mag_model_hsc-i": 27.0,
        "mag_model_hsc-z": 26.5,
        "mag_model_hsc-y": 26.0,
        "mag_model_hst-f814w": 27.0,
        "mag_model_f115w": 27.0,
        "mag_model_f150w": 27.0,
        "mag_model_f277w": 27.0,
        "mag_model_f444w": 27.0,
        "mag_model_f770w": 27.0,
    }

    filter_max = [0] * len(cosmos_web_mag_colnames)

    for filter_name, mmax in filter_thresholds.items():
        if filter_name in cosmos_web_mag_colnames:
            i = cosmos_web_mag_colnames.index(filter_name)
            filter_max[i] = mmax

    mask_mi_drop = (
        (cosmos_web_subset[cosmos_web_mag_colnames[mi]] > filter_max[mi])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi1]] < filter_max[mi1])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi2]] < filter_max[mi2])
    )

    mask_m1_drop = (
        (cosmos_web_subset[cosmos_web_mag_colnames[mi]] < filter_max[mi])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi1]] > filter_max[mi1])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi2]] < filter_max[mi2])
    )

    mask_m2_drop = (
        (cosmos_web_subset[cosmos_web_mag_colnames[mi]] < filter_max[mi])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi1]] < filter_max[mi1])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi2]] > filter_max[mi2])
    )

    mask_mi_m1_drop = (
        (cosmos_web_subset[cosmos_web_mag_colnames[mi]] > filter_max[mi])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi1]] > filter_max[mi1])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi2]] < filter_max[mi2])
    )

    mask_mi_m2_drop = (
        (cosmos_web_subset[cosmos_web_mag_colnames[mi]] > filter_max[mi])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi1]] < filter_max[mi1])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi2]] > filter_max[mi2])
    )

    mask_m1_m2_drop = (
        (cosmos_web_subset[cosmos_web_mag_colnames[mi]] < filter_max[mi])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi1]] > filter_max[mi1])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi2]] > filter_max[mi2])
    )

    mask_mi_m1_m2_drop = (
        (cosmos_web_subset[cosmos_web_mag_colnames[mi]] > filter_max[mi])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi1]] > filter_max[mi1])
        & (cosmos_web_subset[cosmos_web_mag_colnames[mi2]] > filter_max[mi2])
    )

    cosmos_web_4d_data[mask_mi_drop, 0] = 35.0
    cosmos_web_4d_data[mask_mi_drop, 2] = -10.0
    cosmos_web_4d_data[mask_mi_drop, 3] = -10.0

    cosmos_web_4d_data[mask_m1_drop, 3] = 10.0

    cosmos_web_4d_data[mask_m2_drop, 2] = 10.0

    cosmos_web_4d_data[mask_mi_m1_drop, 0] = 45.0
    cosmos_web_4d_data[mask_mi_m1_drop, 3] = 20.0

    cosmos_web_4d_data[mask_mi_m2_drop, 0] = 55.0
    cosmos_web_4d_data[mask_mi_m2_drop, 2] = 20.0

    cosmos_web_4d_data[mask_m1_m2_drop, 2] = 30.0
    cosmos_web_4d_data[mask_m1_m2_drop, 3] = 30.0

    cosmos_web_4d_data[mask_mi_m1_m2_drop, 0] = 65.0
    cosmos_web_4d_data[mask_mi_m1_m2_drop, 2] = 40.0
    cosmos_web_4d_data[mask_mi_m1_m2_drop, 3] = 40.0

    return cosmos_web_4d_data


@partial(jax.jit, static_argnames=("mi", "mi2", "mi1"))
def assign_dropout_values_CW_model(
    phot_info_data,
    mi,
    mi2,
    mi1,
    phot_info,
):

    filter_thresholds = {
        "chft_u": 27.0,
        "hsc_g": 27.0,
        "hsc_r": 27.0,
        "hsc_i": 27.0,
        "hsc_z": 26.5,
        "hsc_y": 26.0,
        "hst_f814w": 27.0,
        "nircam_f115w": 27.0,
        "nircam_f150w": 27.0,
        "nircam_f277w": 27.0,
        "nircam_f444w": 27.0,
        "miri_f770w": 27.0,
    }

    filter_max = [0] * len(config.cosmos_web_filters_to_use)

    for filter_name, mmax in filter_thresholds.items():
        if filter_name in config.cosmos_web_filters_to_use:
            i = config.cosmos_web_filters_to_use.index(filter_name)
            filter_max[i] = mmax

    obs_mags = jnp.stack(phot_info["obs_mags"], axis=0)

    mask_mi_drop = (
        (obs_mags[:, mi] > filter_max[mi])
        & (obs_mags[:, mi1] < filter_max[mi1])
        & (obs_mags[:, mi2] < filter_max[mi2])
    )

    mask_m1_drop = (
        (obs_mags[:, mi] < filter_max[mi])
        & (obs_mags[:, mi1] > filter_max[mi1])
        & (obs_mags[:, mi2] < filter_max[mi2])
    )

    mask_m2_drop = (
        (obs_mags[:, mi] < filter_max[mi])
        & (obs_mags[:, mi1] < filter_max[mi1])
        & (obs_mags[:, mi2] > filter_max[mi2])
    )

    mask_mi_m1_drop = (
        (obs_mags[:, mi] > filter_max[mi])
        & (obs_mags[:, mi1] > filter_max[mi1])
        & (obs_mags[:, mi2] < filter_max[mi2])
    )

    mask_mi_m2_drop = (
        (obs_mags[:, mi] > filter_max[mi])
        & (obs_mags[:, mi1] < filter_max[mi1])
        & (obs_mags[:, mi2] > filter_max[mi2])
    )

    mask_m1_m2_drop = (
        (obs_mags[:, mi] < filter_max[mi])
        & (obs_mags[:, mi1] > filter_max[mi1])
        & (obs_mags[:, mi2] > filter_max[mi2])
    )

    mask_mi_m1_m2_drop = (
        (obs_mags[:, mi] > filter_max[mi])
        & (obs_mags[:, mi1] > filter_max[mi1])
        & (obs_mags[:, mi2] > filter_max[mi2])
    )

    phot_info_data = phot_info_data.at[:, 0].set(
        jnp.where(mask_mi_drop, 35.0, phot_info_data[:, 0])
    )
    phot_info_data = phot_info_data.at[:, 2].set(
        jnp.where(mask_mi_drop, -10.0, phot_info_data[:, 2])
    )
    phot_info_data = phot_info_data.at[:, 3].set(
        jnp.where(mask_mi_drop, -10.0, phot_info_data[:, 3])
    )

    phot_info_data = phot_info_data.at[:, 3].set(
        jnp.where(mask_m1_drop, 10.0, phot_info_data[:, 3])
    )

    phot_info_data = phot_info_data.at[:, 2].set(
        jnp.where(mask_m2_drop, 10.0, phot_info_data[:, 2])
    )

    phot_info_data = phot_info_data.at[:, 0].set(
        jnp.where(mask_mi_m1_drop, 45.0, phot_info_data[:, 0])
    )
    phot_info_data = phot_info_data.at[:, 3].set(
        jnp.where(mask_mi_m1_drop, 20.0, phot_info_data[:, 3])
    )

    phot_info_data = phot_info_data.at[:, 0].set(
        jnp.where(mask_mi_m2_drop, 55.0, phot_info_data[:, 0])
    )
    phot_info_data = phot_info_data.at[:, 2].set(
        jnp.where(mask_mi_m2_drop, 20.0, phot_info_data[:, 2])
    )

    phot_info_data = phot_info_data.at[:, 2].set(
        jnp.where(mask_m1_m2_drop, 30.0, phot_info_data[:, 2])
    )
    phot_info_data = phot_info_data.at[:, 3].set(
        jnp.where(mask_m1_m2_drop, 30.0, phot_info_data[:, 3])
    )

    phot_info_data = phot_info_data.at[:, 0].set(
        jnp.where(mask_mi_m1_m2_drop, 65.0, phot_info_data[:, 0])
    )
    phot_info_data = phot_info_data.at[:, 2].set(
        jnp.where(mask_mi_m1_m2_drop, 40.0, phot_info_data[:, 2])
    )
    phot_info_data = phot_info_data.at[:, 3].set(
        jnp.where(mask_mi_m1_m2_drop, 40.0, phot_info_data[:, 3])
    )

    return phot_info_data


def bin_cosmos_web_4d_data(c4d_data_cut, c4d_data_dropout, N_mag_bins, N_color_bins):

    width_0 = (c4d_data_cut[:, 0].max() - c4d_data_cut[:, 0].min()) / N_mag_bins
    width_1 = (c4d_data_cut[:, 1].max() - c4d_data_cut[:, 1].min()) / N_mag_bins
    width_2 = (c4d_data_cut[:, 2].max() - c4d_data_cut[:, 2].min()) / N_color_bins
    width_3 = (c4d_data_cut[:, 3].max() - c4d_data_cut[:, 3].min()) / N_color_bins

    edges0 = np.arange(
        c4d_data_dropout[:, 0].min(),
        c4d_data_dropout[:, 0].max() + width_0,
        width_0,
    )
    edges1 = np.arange(
        c4d_data_dropout[:, 1].min(),
        c4d_data_dropout[:, 1].max() + width_1,
        width_1,
    )
    edges2 = np.arange(
        c4d_data_dropout[:, 2].min(),
        c4d_data_dropout[:, 2].max() + width_2,
        width_2,
    )
    edges3 = np.arange(
        c4d_data_dropout[:, 3].min(),
        c4d_data_dropout[:, 3].max() + width_3,
        width_3,
    )
    bins = (edges0, edges1, edges2, edges3)
    return bins
