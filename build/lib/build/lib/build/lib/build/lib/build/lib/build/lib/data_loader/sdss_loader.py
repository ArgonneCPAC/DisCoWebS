"""
Author: Natália Villa Nova Rodrigues
Module to load and clean SDSS data.
Adapted from diffsky cosmos data loader:
https://github.com/ArgonneCPAC/diffsky/blob/main/diffsky/data_loaders/cosmos20_loader.py#L167

"""

import os
import numpy as np
import jax.numpy as jnp
from pathlib import Path

try:
    from astropy.table import Table

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


Z_MIN, Z_MAX = 0.02, 0.20
MAGR_THRESH = 17.7
SDSS_MAG_NAMES = ["modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z"]

NANFILL = -999.0

SDSS_BASENAME = "Skyserver_SQL1_23_2026 12_38_32 AM.csv"

SKY_AREA = 8000  # square degrees


def get_sdss_mag_arr_from_cat(sdss_cat):
    """
    Receives astropy catalog and return magnitudes array
    :param sdss_cat:
    :return:
    """
    # Get magnitudes
    sdss_cat = sdss_cat[SDSS_MAG_NAMES]

    # TODO: find better way to convert astropy.Table into np.array
    return sdss_cat.to_pandas().to_numpy()


def load_sdss_wrapper_get_mags_arr(drn=None, bn=SDSS_BASENAME):
    """
    Auxiliary function to load SDSS data + apply all cuts + return the magnitudes as a numpy.array.
    :param drn:
    :param bn:
    :return:
    """
    # Get sdss astropy catalog with all cuts applied
    sdss = load_sdss_wrapper(drn, bn)
    # Get magnitudes and convert into numpy.array
    return get_sdss_mag_arr_from_cat(sdss)


def load_sdss_wrapper(drn=None, bn=SDSS_BASENAME):
    """
    Auxiliary function to load SDSS data, apply all cuts.
    :param drn:
    :param bn:
    :return:
    """

    sdss = load_sdss_cat(drn, bn)

    # Mask out galaxies with NaN or inf in any of our target columns
    sdss = apply_nan_cuts(sdss)

    # Mask out galaxies beyond our completeness cuts
    msk_is_complete = get_is_complete_mask(sdss)
    sdss = sdss[msk_is_complete]

    # Mask out galaxies that are color outliers
    msk_is_not_outlier = get_color_outlier_mask(sdss, SDSS_MAG_NAMES)
    sdss = sdss[msk_is_not_outlier]

    # Apply RA cut -- to make sky area computation easier
    sdss = apply_ra_cuts(sdss)

    return sdss


def load_sdss_cat(
    drn=None,
    bn=SDSS_BASENAME,
    apply_cuts=True,
    mag_lo=5.0,
    mag_hi=30.0,
):
    if not HAS_ASTROPY:
        raise ImportError("Must have astropy installed to use sdss_loader.py")
    if not HAS_PANDAS:
        raise ImportError("Must have pandas installed to use sdss_loader.py")

    if drn is None:
        try:
            drn = os.environ["SDSS_DRN"]
        except KeyError:
            msg = "Must set environment variable SDSS_DRN or pass drn argument"
            raise KeyError(msg)

    if drn.startswith("~"):
        drn = str(Path(drn).expanduser())

    # Load csv file
    fn = os.path.join(drn, bn)
    df = pd.read_csv(fn, skiprows=1)
    # Convert GALAXY string from spec. classification column to int(1) and all other classes to int(0)
    df["class"] = df["class"].map({"GALAXY": 1}).fillna(0).astype(int)
    # Convert pandas.DataFrame into astropy.Table
    cat = Table.from_pandas(df)

    if apply_cuts:
        cat_out = Table()
        cuts = []
        # Select galaxies
        sel_galaxies = np.array(cat["class"] == 1).astype(bool)
        cuts.append(sel_galaxies)

        # Select apparent magnitude columns
        mag_keys = [key for key in cat.keys() if "modelMag_" in key]
        for key in mag_keys:
            # Find NaN and replace with NANFILL value
            x = np.nan_to_num(
                cat[key], copy=True, nan=NANFILL, posinf=NANFILL, neginf=NANFILL
            )
            key_finite_msk = np.isfinite(x == NANFILL)

            cuts.append(key_finite_msk)
            # Select galaxies within the magnitude range
            cuts.append(x > mag_lo)
            cuts.append(x < mag_hi)

        msk = np.prod(cuts, axis=0).astype(bool)
        for key in cat.keys():
            cat_out[key] = jnp.array(cat[key][msk])

        return cat_out

    else:
        return cat


def apply_nan_cuts(sdss, mag_names=SDSS_MAG_NAMES):
    """Remove any galaxy with a NaN in any column storing a target magnitude
    Function adapted from diffsky cosmos data loader.

    Parameters
    ----------
    sdss : astropy Table

    mag_names : list of strings

    Returns
    -------
    sdss : astropy Table
        Catalog after applying a NaN cut on `z` and any colname in mag_names

    """
    msk_has_nan = np.isnan(sdss["z"])
    for name in mag_names:
        x = np.nan_to_num(
            sdss[name], copy=True, nan=NANFILL, posinf=NANFILL, neginf=NANFILL
        )
        msk_has_nan = msk_has_nan | (x == NANFILL)

    sdss = sdss[~msk_has_nan]
    return sdss


def get_is_complete_mask(sdss, z_min=Z_MIN, z_max=Z_MAX, magr_thresh=MAGR_THRESH):
    """Compute mask to define the redshift and r-mag threshold for our target data
    Function adapted from diffsky cosmos data loader.

    Parameters
    ----------
    sdss : astropy Table

    z_min, z_max : float
        Galaxies outside this range will be excluded

    magr_thresh : float
        Galaxies fainter than this apparent magnitude will be excluded

    Returns
    -------
    msk_is_complete : array, dtype bool
        Boolean mask defining galaxies that pass the completeness cut

    """
    msk_redshift = (sdss["z"] > z_min) & (sdss["z"] < z_max)
    msk_r_thresh = sdss["modelMag_r"] <= magr_thresh
    msk_is_complete = msk_redshift & msk_r_thresh
    return msk_is_complete


def get_color_outlier_mask(sdss, mag_names, p_cut=0.5):
    """Compute mask to define extreme outliers in color space
    Function adapted from diffsky cosmos data loader.

    Parameters
    ----------
    sdss : astropy Table

    mag_names : list of strings
        Column names defining the colors for which outliers will be excluded

    p_cut : float, optional
        Value in the range [0, 100] defining a percentile cut

    Returns
    -------
    msk_is_not_outlier : array, dtype bool
        Boolean mask defining galaxies that pass the outlier cut

    """
    msk_is_outlier = np.zeros(len(sdss)).astype(bool)
    for name0, name1 in zip(mag_names[0:], mag_names[1:]):
        c0 = sdss[name0]
        c1 = sdss[name1]
        color = c0 - c1
        lo, hi = np.percentile(color, (p_cut, 100.0 - p_cut))
        msk_is_outlier = msk_is_outlier | (color < lo) | (color > hi)

    msk_is_not_outlier = ~msk_is_outlier

    return msk_is_not_outlier


def apply_ra_cuts(sdss, ra_min=100, ra_max=250):
    return sdss[(sdss["ra"] > ra_min) & (sdss["ra"] < ra_max)]
