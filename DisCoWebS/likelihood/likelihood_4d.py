import numpy as np
import jax.numpy as jnp
from jax import random as jran
from diffsky.experimental import lightcone_generators as lcg
from diffsky.experimental.kernels import gd_phot_kernels_merging as gpkm
from dsps.cosmology import DEFAULT_COSMOLOGY
from diffstar.defaults import FB
from diffsky import signdhist_lomem
from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm
from jax import jit as jjit
from functools import partial
from .likelihood_kernel import (
    mse_loss,
    log_mse_loss,
    ln_poisson_loss,
)
from ..data_loader.cosmos_web_loader import (
    impose_cosmos_web_mag_cut,
    assign_dropout_values_CW_data,
    assign_dropout_values_CW_model,
)
from DisCoWebS.modelling.igm import igm_attenuation
from ..utils.hist import sparse_histogramdd_large


def bin_cosmos_data_m_i_c1_c2(
    cosmos,
    cosmos_mag_colnames,
    N_z_bins,
    N_host_min,
    N_host_max,
    params_b,
    n_z_phot_table,
    ssp_data,
    tcurves,
    ran_key,
    N_mag_bins,
    N_color_bins,
    ndsig_by_dbin,
):
    sky_area = 1.21  # c20.SKY_AREA
    z_bins = np.linspace(
        np.min(cosmos["photoz"]), np.max(cosmos["photoz"]), N_z_bins + 1
    )
    i_i = cosmos_mag_colnames.index("HSC_i_MAG")

    lc_data_all = []
    M_c_data_all = []
    ndsig_M_c_pred_all = []
    M_c_min_all = []
    M_c_max_all = []
    n_gal_all = []

    for zi in range(N_z_bins):
        num_halos = N_host_min + zi * (N_host_max - N_host_min) // (N_z_bins - 1)
        z_min, z_max = z_bins[zi], z_bins[zi + 1]
        lgmp_min, lgmp_max = 10.0 + zi * 1.0 / N_z_bins, 14.8

        z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

        halo_lc_data = (num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area)
        phot_data = (ssp_data, tcurves, z_phot_table)

        ran_key, lc_halo_key = jran.split(ran_key, 2)
        args = (lc_halo_key, *halo_lc_data, *phot_data)

        lc_data = lcg.weighted_lc_photdata(*args)
        lc_data_all.append(lc_data)

        n_gal = lc_data.cen_weight * lc_data.sat_weight
        n_gal_all.append(n_gal)

        ran_key, sed_key = jran.split(ran_key, 2)

        phot_info, phot_randoms, merging_randoms = gpkm._mc_phot_kern_merging(
            sed_key,
            lc_data.z_obs,
            lc_data.t_obs,
            lc_data.mah_params,
            lc_data.ssp_data,
            lc_data.precomputed_ssp_mag_table,
            lc_data.z_phot_table,
            lc_data.wave_eff_table,
            params_b.diffstarpop_params,
            params_b.mzr_params,
            params_b.spspop_params,
            params_b.scatter_params,
            params_b.ssperr_params,
            params_b.merging_params,
            DEFAULT_COSMOLOGY,
            FB,
            lc_data.logmp_infall,
            lc_data.logmhost_infall,
            lc_data.t_infall,
            lc_data.is_central,
            lc_data.sat_weight,
            lc_data.halo_indx,
            mc_merge=0,
        )

        cosmos_subset = cosmos[(cosmos["photoz"] >= z_min) & (cosmos["photoz"] < z_max)]

        cosmos_subset_cut = cosmos[
            (cosmos["photoz"] >= z_min) & (cosmos["photoz"] < z_max)
        ]

        n_mag = len(cosmos_mag_colnames)
        cosmos_subset_cut = cosmos_subset_cut[
            (cosmos_subset_cut[cosmos_mag_colnames[i_i]] < 25.0)
        ]

        for mi in range(n_mag):
            if mi <= n_mag - 2:
                mi2 = mi + 1
            else:
                mi2 = 0
            if mi >= 1:
                mi1 = mi - 1
            else:
                mi1 = n_mag - 1

            cosmos_data_cut = np.vstack(
                (
                    cosmos_subset_cut[cosmos_mag_colnames[mi]],
                    cosmos_subset_cut["HSC_i_MAG"],
                    cosmos_subset_cut[cosmos_mag_colnames[mi2]]
                    - cosmos_subset_cut[cosmos_mag_colnames[mi]],
                    cosmos_subset_cut[cosmos_mag_colnames[mi1]]
                    - cosmos_subset_cut[cosmos_mag_colnames[mi]],
                )
            ).T

            bin_widths = [
                (cosmos_data_cut[:, 0].max() - cosmos_data_cut[:, 0].min())
                / N_mag_bins,
                (cosmos_data_cut[:, 1].max() - cosmos_data_cut[:, 1].min())
                / N_mag_bins,
                (cosmos_data_cut[:, 2].max() - cosmos_data_cut[:, 2].min())
                / N_color_bins,
                (cosmos_data_cut[:, 3].max() - cosmos_data_cut[:, 3].min())
                / N_color_bins,
            ]

            Hist_nD1, edges, occupied_bins = sparse_histogramdd_large(
                cosmos_data_cut,
                bin_widths=bin_widths,
                chunk_size=500_000,
            )

            phot_info_data = np.vstack(
                (
                    phot_info.obs_mags_weighted[:, mi],
                    phot_info.obs_mags_weighted[:, i_i],
                    phot_info.obs_mags_weighted[:, mi2]
                    - phot_info.obs_mags_weighted[:, mi],
                    phot_info.obs_mags_weighted[:, mi1]
                    - phot_info.obs_mags_weighted[:, mi],
                )
            ).T

            Hist_nD2, bins, occupied_bins = sparse_histogramdd_large(
                phot_info_data,
                edges=edges,
                occupied_bins=occupied_bins,
                chunk_size=500_000,
            )

            non_zero_indices = np.where((Hist_nD1 >= 1.0) & (Hist_nD2 >= 1.0))
            if len(non_zero_indices[0]) <= 1:
                raise ValueError(
                    f"Data and model histograms don't overlap for z bin {zi}."
                )

            switch = 1
            for i in range(len(non_zero_indices[0])):
                if switch == 1:
                    _min_0 = np.array(
                        [bins[i][0][0], bins[i][1][0], bins[i][2][0], bins[i][3][0]]
                    )
                    _max_0 = np.array(
                        [bins[i][0][1], bins[i][1][1], bins[i][2][1], bins[i][3][1]]
                    )
                    switch = 0
                else:
                    _min_j = np.array(
                        [bins[i][0][0], bins[i][1][0], bins[i][2][0], bins[i][3][0]]
                    )
                    _max_j = np.array(
                        [bins[i][0][1], bins[i][1][1], bins[i][2][1], bins[i][3][1]]
                    )
                    _min_0 = np.vstack([_min_0, _min_j])
                    _max_0 = np.vstack([_max_0, _max_j])

            M_c_min_ = jnp.array(_min_0)
            M_c_max_ = jnp.array(_max_0)

            ndsig_M_c_data_ = (M_c_max_ - M_c_min_) * ndsig_by_dbin

            ndsig_M_c_pred_ = (M_c_max_ - M_c_min_) * ndsig_by_dbin

            M_c_data_ = signdhist_lomem.nnsig_ndhist(
                np.vstack(
                    (
                        cosmos_subset[cosmos_mag_colnames[mi]],
                        cosmos_subset["HSC_i_MAG"],
                        cosmos_subset[cosmos_mag_colnames[mi2]]
                        - cosmos_subset[cosmos_mag_colnames[mi]],
                        cosmos_subset[cosmos_mag_colnames[mi1]]
                        - cosmos_subset[cosmos_mag_colnames[mi]],
                    )
                ).T,
                ndsig_M_c_data_,
                M_c_min_,
                M_c_max_,
            )

            M_c_data_all.append(M_c_data_)

            ndsig_M_c_pred_all.append(ndsig_M_c_pred_)
            M_c_min_all.append(M_c_min_)
            M_c_max_all.append(M_c_max_)

    return (
        ran_key,
        lc_data_all,
        M_c_data_all,
        ndsig_M_c_pred_all,
        M_c_min_all,
        M_c_max_all,
        n_gal_all,
    )


def bin_sdss_data_m_i_c1_c2(
    sdss,
    sdss_mag_colnames,
    N_z_bins,
    N_host_min,
    N_host_max,
    params_b,
    n_z_phot_table,
    ssp_data,
    tcurves,
    ran_key,
    N_mag_bins,
    N_color_bins,
    ndsig_by_dbin,
):
    sky_area = 8000.0  # sdss.SKY_AREA
    z_bins = np.linspace(np.min(sdss["z"]), np.max(sdss["z"]), N_z_bins + 1)
    i_i = sdss_mag_colnames.index("modelMag_r")

    lc_data_all = []
    M_c_data_all = []
    ndsig_M_c_pred_all = []
    M_c_min_all = []
    M_c_max_all = []
    n_gal_all = []

    for zi in range(N_z_bins):
        num_halos = N_host_min + zi * (N_host_max - N_host_min) // (N_z_bins - 1)
        z_min, z_max = z_bins[zi], z_bins[zi + 1]
        lgmp_min, lgmp_max = 10.0 + zi * 1.5 / N_z_bins, 15.0

        z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

        halo_lc_data = (num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area)
        phot_data = (ssp_data, tcurves, z_phot_table)

        ran_key, lc_halo_key = jran.split(ran_key, 2)
        args = (lc_halo_key, *halo_lc_data, *phot_data)

        lc_data = lcg.weighted_lc_photdata(*args)
        lc_data_all.append(lc_data)

        n_gal = lc_data.cen_weight * lc_data.sat_weight
        n_gal_all.append(n_gal)

        ran_key, sed_key = jran.split(ran_key, 2)

        phot_info, phot_randoms, merging_randoms = gpkm._mc_phot_kern_merging(
            sed_key,
            lc_data.z_obs,
            lc_data.t_obs,
            lc_data.mah_params,
            lc_data.ssp_data,
            lc_data.precomputed_ssp_mag_table,
            lc_data.z_phot_table,
            lc_data.wave_eff_table,
            params_b.diffstarpop_params,
            params_b.mzr_params,
            params_b.spspop_params,
            params_b.scatter_params,
            params_b.ssperr_params,
            params_b.merging_params,
            DEFAULT_COSMOLOGY,
            FB,
            lc_data.logmp_infall,
            lc_data.logmhost_infall,
            lc_data.t_infall,
            lc_data.is_central,
            lc_data.sat_weight,
            lc_data.halo_indx,
            mc_merge=0,
        )

        sdss_subset = sdss[(sdss["z"] >= z_min) & (sdss["z"] < z_max)]

        sdss_subset_cut = sdss[(sdss["z"] >= z_min) & (sdss["z"] < z_max)]

        n_mag = len(sdss_mag_colnames)
        sdss_subset_cut = sdss_subset_cut[
            (sdss_subset_cut[sdss_mag_colnames[i_i]] < 17.5)
        ]

        for mi in range(n_mag):
            if mi <= n_mag - 2:
                mi2 = mi + 1
            else:
                mi2 = 0
            if mi >= 1:
                mi1 = mi - 1
            else:
                mi1 = n_mag - 1

            sdss_data_cut = np.vstack(
                (
                    sdss_subset_cut[sdss_mag_colnames[mi]],
                    sdss_subset_cut["modelMag_r"],
                    sdss_subset_cut[sdss_mag_colnames[mi2]]
                    - sdss_subset_cut[sdss_mag_colnames[mi]],
                    sdss_subset_cut[sdss_mag_colnames[mi1]]
                    - sdss_subset_cut[sdss_mag_colnames[mi]],
                )
            ).T

            bin_widths = [
                (sdss_data_cut[:, 0].max() - sdss_data_cut[:, 0].min()) / N_mag_bins,
                (sdss_data_cut[:, 1].max() - sdss_data_cut[:, 1].min()) / N_mag_bins,
                (sdss_data_cut[:, 2].max() - sdss_data_cut[:, 2].min()) / N_color_bins,
                (sdss_data_cut[:, 3].max() - sdss_data_cut[:, 3].min()) / N_color_bins,
            ]

            Hist_nD1, edges, occupied_bins = sparse_histogramdd_large(
                sdss_data_cut,
                bin_widths=bin_widths,
                chunk_size=500_000,
            )

            phot_info_data = np.vstack(
                (
                    phot_info.obs_mags_weighted[:, mi],
                    phot_info.obs_mags_weighted[:, i_i],
                    phot_info.obs_mags_weighted[:, mi2]
                    - phot_info.obs_mags_weighted[:, mi],
                    phot_info.obs_mags_weighted[:, mi1]
                    - phot_info.obs_mags_weighted[:, mi],
                )
            ).T

            Hist_nD2, bins, occupied_bins = sparse_histogramdd_large(
                phot_info_data,
                edges=edges,
                occupied_bins=occupied_bins,
                chunk_size=500_000,
            )

            non_zero_indices = np.where((Hist_nD1 >= 1.0) & (Hist_nD2 >= 1.0))
            if len(non_zero_indices[0]) <= 1:
                raise ValueError(
                    f"Data and model histograms don't overlap for z bin {zi}."
                )

            switch = 1
            for i in range(len(non_zero_indices[0])):
                if switch == 1:
                    _min_0 = np.array(
                        [bins[i][0][0], bins[i][1][0], bins[i][2][0], bins[i][3][0]]
                    )
                    _max_0 = np.array(
                        [bins[i][0][1], bins[i][1][1], bins[i][2][1], bins[i][3][1]]
                    )
                    switch = 0
                else:
                    _min_j = np.array(
                        [bins[i][0][0], bins[i][1][0], bins[i][2][0], bins[i][3][0]]
                    )
                    _max_j = np.array(
                        [bins[i][0][1], bins[i][1][1], bins[i][2][1], bins[i][3][1]]
                    )
                    _min_0 = np.vstack([_min_0, _min_j])
                    _max_0 = np.vstack([_max_0, _max_j])

            M_c_min_ = jnp.array(_min_0)
            M_c_max_ = jnp.array(_max_0)

            ndsig_M_c_data_ = (M_c_max_ - M_c_min_) * ndsig_by_dbin

            ndsig_M_c_pred_ = (M_c_max_ - M_c_min_) * ndsig_by_dbin

            M_c_data_ = signdhist_lomem.nnsig_ndhist(
                np.vstack(
                    (
                        sdss_subset[sdss_mag_colnames[mi]],
                        sdss_subset["modelMag_r"],
                        sdss_subset[sdss_mag_colnames[mi2]]
                        - sdss_subset[sdss_mag_colnames[mi]],
                        sdss_subset[sdss_mag_colnames[mi1]]
                        - sdss_subset[sdss_mag_colnames[mi]],
                    )
                ).T,
                ndsig_M_c_data_,
                M_c_min_,
                M_c_max_,
            )

            M_c_data_all.append(M_c_data_)

            ndsig_M_c_pred_all.append(ndsig_M_c_pred_)
            M_c_min_all.append(M_c_min_)
            M_c_max_all.append(M_c_max_)

    return (
        ran_key,
        lc_data_all,
        M_c_data_all,
        ndsig_M_c_pred_all,
        M_c_min_all,
        M_c_max_all,
        n_gal_all,
    )


def bin_cosmos_web_data_m_i_c1_c2(
    cosmos_web,
    cosmos_web_mag_colnames,
    N_z_bins,
    N_host_min,
    N_host_max,
    params_b,
    n_z_phot_table,
    ssp_data,
    tcurves,
    ran_key,
    N_mag_bins,
    N_color_bins,
    ndsig_by_dbin,
):
    sky_area = 0.433  # cosmos_web_without_MIRI.SKY_AREA
    z_bins = np.linspace(
        np.min(cosmos_web["zfinal"]), np.max(cosmos_web["zfinal"]), N_z_bins + 1
    )
    i_i = cosmos_web_mag_colnames.index("mag_model_f444w")

    lc_data_all = []
    M_c_data_all = []
    ndsig_M_c_pred_all = []
    M_c_min_all = []
    M_c_max_all = []
    n_gal_all = []

    for zi in range(N_z_bins):
        num_halos = N_host_min + zi * (N_host_max - N_host_min) // (N_z_bins - 1)
        z_min, z_max = z_bins[zi], z_bins[zi + 1]
        lgmp_min = 8.0 + zi * 1.0 / N_z_bins
        lgmp_max = 13.7 + zi * 0.5 / N_z_bins

        z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

        halo_lc_data = (num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area)
        phot_data = (ssp_data, tcurves, z_phot_table)

        ran_key, lc_halo_key = jran.split(ran_key, 2)
        args = (lc_halo_key, *halo_lc_data, *phot_data)

        lc_data = lcg.weighted_lc_photdata(*args)
        lc_data_all.append(lc_data)

        n_gal = lc_data.cen_weight * lc_data.sat_weight
        n_gal_all.append(n_gal)

        ran_key, sed_key = jran.split(ran_key, 2)

        phot_info, phot_randoms, merging_randoms = gpkm._mc_phot_kern_merging(
            sed_key,
            lc_data.z_obs,
            lc_data.t_obs,
            lc_data.mah_params,
            lc_data.ssp_data,
            lc_data.precomputed_ssp_mag_table,
            lc_data.z_phot_table,
            lc_data.wave_eff_table,
            params_b.diffstarpop_params,
            params_b.mzr_params,
            params_b.spspop_params,
            params_b.scatter_params,
            params_b.ssperr_params,
            params_b.merging_params,
            DEFAULT_COSMOLOGY,
            FB,
            lc_data.logmp_infall,
            lc_data.logmhost_infall,
            lc_data.t_infall,
            lc_data.is_central,
            lc_data.sat_weight,
            lc_data.halo_indx,
            mc_merge=0,
        )

        cosmos_web_subset = cosmos_web[
            (cosmos_web["zfinal"] >= z_min)
            & (cosmos_web["zfinal"] < z_max)
            & (cosmos_web[cosmos_web_mag_colnames[i_i]] < 27.0)
        ]

        cosmos_web_subset_cut = impose_cosmos_web_mag_cut(
            cosmos_web_subset, cosmos_web_mag_colnames
        )

        n_mag = len(cosmos_web_mag_colnames)

        for mi in range(n_mag):
            if mi <= n_mag - 2:
                mi2 = mi + 1
            else:
                mi2 = 0
            if mi >= 1:
                mi1 = mi - 1
            else:
                mi1 = n_mag - 1

            cosmos_web_4d_data = np.vstack(
                (
                    cosmos_web_subset[cosmos_web_mag_colnames[mi]],
                    cosmos_web_subset[cosmos_web_mag_colnames[i_i]],
                    cosmos_web_subset[cosmos_web_mag_colnames[mi2]]
                    - cosmos_web_subset[cosmos_web_mag_colnames[mi]],
                    cosmos_web_subset[cosmos_web_mag_colnames[mi1]]
                    - cosmos_web_subset[cosmos_web_mag_colnames[mi]],
                )
            ).T

            cosmos_web_4d_data_dropout = assign_dropout_values_CW_data(
                cosmos_web_4d_data,
                cosmos_web_mag_colnames,
                mi,
                mi2,
                mi1,
                cosmos_web_subset,
            )

            cosmos_web_4d_data_cut = np.vstack(
                (
                    cosmos_web_subset_cut[cosmos_web_mag_colnames[mi]],
                    cosmos_web_subset_cut[cosmos_web_mag_colnames[i_i]],
                    cosmos_web_subset_cut[cosmos_web_mag_colnames[mi2]]
                    - cosmos_web_subset_cut[cosmos_web_mag_colnames[mi]],
                    cosmos_web_subset_cut[cosmos_web_mag_colnames[mi1]]
                    - cosmos_web_subset_cut[cosmos_web_mag_colnames[mi]],
                )
            ).T

            bin_widths = [
                (
                    cosmos_web_4d_data_cut[:, 0].max()
                    - cosmos_web_4d_data_cut[:, 0].min()
                )
                / N_mag_bins,
                (
                    cosmos_web_4d_data_cut[:, 1].max()
                    - cosmos_web_4d_data_cut[:, 1].min()
                )
                / N_mag_bins,
                (
                    cosmos_web_4d_data_cut[:, 2].max()
                    - cosmos_web_4d_data_cut[:, 2].min()
                )
                / N_color_bins,
                (
                    cosmos_web_4d_data_cut[:, 3].max()
                    - cosmos_web_4d_data_cut[:, 3].min()
                )
                / N_color_bins,
            ]

            Hist_nD1, edges, occupied_bins = sparse_histogramdd_large(
                cosmos_web_4d_data_dropout,
                bin_widths=bin_widths,
                chunk_size=500_000,
            )

            phot_info_obs_mags_wIGM = igm_attenuation(
                phot_info.obs_mags_weighted,
                lc_data.z_obs,
            )

            phot_info_data = jnp.vstack(
                (
                    phot_info_obs_mags_wIGM[:, mi],
                    phot_info_obs_mags_wIGM[:, i_i],
                    phot_info_obs_mags_wIGM[:, mi2] - phot_info_obs_mags_wIGM[:, mi],
                    phot_info_obs_mags_wIGM[:, mi1] - phot_info_obs_mags_wIGM[:, mi],
                )
            ).T

            phot_info_data_dropout = assign_dropout_values_CW_model(
                phot_info_data,
                mi,
                mi2,
                mi1,
                phot_info,
            )

            mask = phot_info_data_dropout[:, i_i] > 27.0
            phot_info_data_dropout = phot_info_data_dropout.at[mask, :].set(999.0)

            Hist_nD2, bins, occupied_bins = sparse_histogramdd_large(
                phot_info_data_dropout,
                edges=edges,
                occupied_bins=occupied_bins,
                chunk_size=500_000,
            )

            non_zero_indices = np.where((Hist_nD1 >= 1.0) & (Hist_nD2 >= 1.0))
            if len(non_zero_indices[0]) <= 1:
                raise ValueError(
                    f"Data and model histograms don't overlap for z bin {zi}."
                )

            switch = 1
            for i in range(len(non_zero_indices[0])):
                if switch == 1:
                    _min_0 = np.array(
                        [bins[i][0][0], bins[i][1][0], bins[i][2][0], bins[i][3][0]]
                    )
                    _max_0 = np.array(
                        [bins[i][0][1], bins[i][1][1], bins[i][2][1], bins[i][3][1]]
                    )
                    switch = 0
                else:
                    _min_j = np.array(
                        [bins[i][0][0], bins[i][1][0], bins[i][2][0], bins[i][3][0]]
                    )
                    _max_j = np.array(
                        [bins[i][0][1], bins[i][1][1], bins[i][2][1], bins[i][3][1]]
                    )
                    _min_0 = np.vstack([_min_0, _min_j])
                    _max_0 = np.vstack([_max_0, _max_j])

            M_c_min_ = jnp.array(_min_0)
            M_c_max_ = jnp.array(_max_0)

            ndsig_M_c_data_ = (M_c_max_ - M_c_min_) * ndsig_by_dbin

            ndsig_M_c_pred_ = (M_c_max_ - M_c_min_) * ndsig_by_dbin

            M_c_data_ = signdhist_lomem.nnsig_ndhist(
                cosmos_web_4d_data_dropout,
                ndsig_M_c_data_,
                M_c_min_,
                M_c_max_,
            )

            M_c_data_all.append(M_c_data_)

            ndsig_M_c_pred_all.append(ndsig_M_c_pred_)
            M_c_min_all.append(M_c_min_)
            M_c_max_all.append(M_c_max_)

    return (
        ran_key,
        lc_data_all,
        M_c_data_all,
        ndsig_M_c_pred_all,
        M_c_min_all,
        M_c_max_all,
        n_gal_all,
    )


@partial(
    jjit, static_argnames=["N_z_bins", "i_i", "n_mag", "mode", "loss_type", "norm"]
)
def m_i_c1_c2_loss(
    params_u,
    sed_key,
    n_gal_all,
    lc_data_all,
    M_c_data_all,
    ndsig_M_c_pred_all,
    M_c_min_all,
    M_c_max_all,
    N_z_bins,
    i_i,
    n_mag,
    params_b_fixed,
    mode="fit_all",
    loss_type="log_mse_loss",
    norm=1.0,
):
    N_z_bins = int(N_z_bins)
    i_i = int(i_i)
    n_mag = int(n_mag)
    mode = str(mode)
    loss_type = str(loss_type)
    norm = float(norm)

    kk = 0
    sum_1 = 0.0

    params_bounded = dpwm.get_param_collection_from_u_param_collection(
        params_u.diffstarpop_u_params,
        params_u.mzr_u_params,
        params_u.spspop_u_params,
        params_u.scatter_u_params,
        params_u.ssperr_u_params,
        params_u.merging_u_params,
    )

    if mode == "fit_diffstarpop":
        params_bounded._replace(
            mzr_params=params_b_fixed.mzr_params,
            spspop_params=params_b_fixed.spspop_params,
            scatter_params=params_b_fixed.scatter_params,
            ssperr_params=params_b_fixed.ssperr_params,
            merging_params=params_b_fixed.merging_params,
        )
    elif mode == "fit_ssperr":
        params_bounded._replace(
            diffstarpop_params=params_b_fixed.diffstarpop_params,
            mzr_params=params_b_fixed.mzr_params,
            spspop_params=params_b_fixed.spspop_params,
            scatter_params=params_b_fixed.scatter_params,
            merging_params=params_b_fixed.merging_params,
        )
    elif mode == "fit_merging":
        params_bounded._replace(
            diffstarpop_params=params_b_fixed.diffstarpop_params,
            mzr_params=params_b_fixed.mzr_params,
            spspop_params=params_b_fixed.spspop_params,
            scatter_params=params_b_fixed.scatter_params,
            ssperr_params=params_b_fixed.ssperr_params,
        )
    elif mode == "fix_diffstarpop":
        params_bounded._replace(diffstarpop_params=params_b_fixed.diffstarpop_params)
    elif mode == "fix_ssperr":
        params_bounded._replace(ssperr_params=params_b_fixed.ssperr_params)
    elif mode == "fix_merging":
        params_bounded._replace(merging_params=params_b_fixed.merging_params)
    elif mode == "fix_diffstarpop_merging":
        params_bounded._replace(
            diffstarpop_params=params_b_fixed.diffstarpop_params,
            merging_params=params_b_fixed.merging_params,
        )
    elif mode == "fix_diffstarpop_ssperr":
        params_bounded._replace(
            diffstarpop_params=params_b_fixed.diffstarpop_params,
            ssperr_params=params_b_fixed.ssperr_params,
        )
    elif mode == "fix_ssperr_merging":
        params_bounded._replace(
            ssperr_params=params_b_fixed.ssperr_params,
            merging_params=params_b_fixed.merging_params,
        )
    elif mode != "fit_all":
        valid_modes = [
            "fit_all",
            "fit_diffstarpop",
            "fit_ssperr",
            "fit_merging",
            "fix_diffstarpop",
            "fix_ssperr",
            "fix_merging",
            "fix_diffstarpop_merging",
            "fix_diffstarpop_ssperr",
            "fix_ssperr_merging",
        ]
        raise ValueError(
            "Invalid mode. Choose from the following:"
            + "\n".join(f" - {m}" for m in valid_modes)
        )

    for zi in range(N_z_bins):
        lc_data = lc_data_all[zi]
        n_gal = n_gal_all[zi]

        phot_info, phot_randoms, merging_randoms = gpkm._mc_phot_kern_merging(
            sed_key,
            lc_data.z_obs,
            lc_data.t_obs,
            lc_data.mah_params,
            lc_data.ssp_data,
            lc_data.precomputed_ssp_mag_table,
            lc_data.z_phot_table,
            lc_data.wave_eff_table,
            params_bounded.diffstarpop_params,
            params_bounded.mzr_params,
            params_bounded.spspop_params,
            params_bounded.scatter_params,
            params_bounded.ssperr_params,
            params_bounded.merging_params,
            DEFAULT_COSMOLOGY,
            FB,
            lc_data.logmp_infall,
            lc_data.logmhost_infall,
            lc_data.t_infall,
            lc_data.is_central,
            lc_data.sat_weight,
            lc_data.halo_indx,
            mc_merge=0,
        )

        for mi in range(n_mag):
            if mi <= n_mag - 2:
                mi2 = mi + 1
            else:
                mi2 = 0
            if mi >= 1:
                mi1 = mi - 1
            else:
                mi1 = n_mag - 1

            M_c_min_ = M_c_min_all[kk]
            M_c_max_ = M_c_max_all[kk]

            ndsig_M_c_pred_ = ndsig_M_c_pred_all[kk]

            M_c_data_ = M_c_data_all[kk]

            M_c_pred_ = signdhist_lomem.nnsig_ndhist_weighted(
                jnp.vstack(
                    (
                        phot_info.obs_mags_weighted[:, mi],
                        phot_info.obs_mags_weighted[:, i_i],
                        phot_info.obs_mags_weighted[:, mi2]
                        - phot_info.obs_mags_weighted[:, mi],
                        phot_info.obs_mags_weighted[:, mi1]
                        - phot_info.obs_mags_weighted[:, mi],
                    )
                ).T,
                ndsig_M_c_pred_,
                n_gal,
                M_c_min_,
                M_c_max_,
            )

            if loss_type == "log_mse_loss":
                eps = 1e-1  # to avoid log(0) issues
                sum_1 = sum_1 + log_mse_loss(M_c_data_, M_c_pred_, eps)
            elif loss_type == "mse_loss":
                sum_1 = sum_1 + mse_loss(M_c_data_, M_c_pred_, eps)
            elif loss_type == "ln_poisson_loss":
                eps = 1e-3  # to avoid log(0) issues
                sum_1 = sum_1 + ln_poisson_loss(M_c_data_, M_c_pred_, eps)
            else:
                valid_modes = [
                    "log_mse_loss",
                    "mse_loss",
                    "ln_poisson_loss",
                ]
                raise ValueError(
                    "Invalid loss_type. Choose from the following:"
                    + "\n".join(f" - {m}" for m in valid_modes)
                )

            kk = kk + 1

    return sum_1 / norm


@partial(
    jjit,
    static_argnames=[
        "N_z_bins",
        "i_i_combined",
        "n_mag_combined",
        "mode",
        "loss_type",
        "norm",
        "data_to_use",
    ],
)
def m_i_c1_c2_loss_multi_data(
    params_u,
    sed_key,
    n_gal_all_combined,
    lc_data_all_combined,
    M_c_data_all_combined,
    ndsig_M_c_pred_all_combined,
    M_c_min_all_combined,
    M_c_max_all_combined,
    N_z_bins,
    i_i_combined,
    n_mag_combined,
    params_b_fixed,
    mode="fit_all",
    loss_type="log_mse_loss",
    norm=1.0,
    data_to_use=("cosmos", "sdss"),
):
    mode = str(mode)
    loss_type = str(loss_type)
    norm = float(norm)
    total_loss = 0.0
    for di in range(len(n_gal_all_combined)):
        n_gal_all = n_gal_all_combined[di]
        lc_data_all = lc_data_all_combined[di]
        M_c_data_all = M_c_data_all_combined[di]
        ndsig_M_c_pred_all = ndsig_M_c_pred_all_combined[di]
        M_c_min_all = M_c_min_all_combined[di]
        M_c_max_all = M_c_max_all_combined[di]
        N_z_bin = int(N_z_bins[di])
        i_i = int(i_i_combined[di])
        n_mag = int(n_mag_combined[di])

        if data_to_use[di] in ["cosmos_web"]:
            loss_di = m_i_c1_c2_loss_cosmos_web(
                params_u,
                sed_key,
                n_gal_all,
                lc_data_all,
                M_c_data_all,
                ndsig_M_c_pred_all,
                M_c_min_all,
                M_c_max_all,
                N_z_bin,
                i_i,
                n_mag,
                params_b_fixed,
                mode=mode,
                loss_type=loss_type,
                norm=norm,
            )
            total_loss = total_loss + loss_di
        else:
            loss_di = m_i_c1_c2_loss(
                params_u,
                sed_key,
                n_gal_all,
                lc_data_all,
                M_c_data_all,
                ndsig_M_c_pred_all,
                M_c_min_all,
                M_c_max_all,
                N_z_bin,
                i_i,
                n_mag,
                params_b_fixed,
                mode=mode,
                loss_type=loss_type,
                norm=norm,
            )
            total_loss = total_loss + loss_di

    return total_loss


@partial(
    jjit, static_argnames=["N_z_bins", "i_i", "n_mag", "mode", "loss_type", "norm"]
)
def m_i_c1_c2_loss_cosmos_web(
    params_u,
    sed_key,
    n_gal_all,
    lc_data_all,
    M_c_data_all,
    ndsig_M_c_pred_all,
    M_c_min_all,
    M_c_max_all,
    N_z_bins,
    i_i,
    n_mag,
    params_b_fixed,
    mode="fit_all",
    loss_type="log_mse_loss",
    norm=1.0,
):
    N_z_bins = int(N_z_bins)
    i_i = int(i_i)
    n_mag = int(n_mag)
    mode = str(mode)
    loss_type = str(loss_type)
    norm = float(norm)

    kk = 0
    sum_1 = 0.0

    params_bounded = dpwm.get_param_collection_from_u_param_collection(
        params_u.diffstarpop_u_params,
        params_u.mzr_u_params,
        params_u.spspop_u_params,
        params_u.scatter_u_params,
        params_u.ssperr_u_params,
        params_u.merging_u_params,
    )

    if mode == "fit_diffstarpop":
        params_bounded._replace(
            mzr_params=params_b_fixed.mzr_params,
            spspop_params=params_b_fixed.spspop_params,
            scatter_params=params_b_fixed.scatter_params,
            ssperr_params=params_b_fixed.ssperr_params,
            merging_params=params_b_fixed.merging_params,
        )
    elif mode == "fit_ssperr":
        params_bounded._replace(
            diffstarpop_params=params_b_fixed.diffstarpop_params,
            mzr_params=params_b_fixed.mzr_params,
            spspop_params=params_b_fixed.spspop_params,
            scatter_params=params_b_fixed.scatter_params,
            merging_params=params_b_fixed.merging_params,
        )
    elif mode == "fit_merging":
        params_bounded._replace(
            diffstarpop_params=params_b_fixed.diffstarpop_params,
            mzr_params=params_b_fixed.mzr_params,
            spspop_params=params_b_fixed.spspop_params,
            scatter_params=params_b_fixed.scatter_params,
            ssperr_params=params_b_fixed.ssperr_params,
        )
    elif mode == "fix_diffstarpop":
        params_bounded._replace(diffstarpop_params=params_b_fixed.diffstarpop_params)
    elif mode == "fix_ssperr":
        params_bounded._replace(ssperr_params=params_b_fixed.ssperr_params)
    elif mode == "fix_merging":
        params_bounded._replace(merging_params=params_b_fixed.merging_params)
    elif mode == "fix_diffstarpop_merging":
        params_bounded._replace(
            diffstarpop_params=params_b_fixed.diffstarpop_params,
            merging_params=params_b_fixed.merging_params,
        )
    elif mode == "fix_diffstarpop_ssperr":
        params_bounded._replace(
            diffstarpop_params=params_b_fixed.diffstarpop_params,
            ssperr_params=params_b_fixed.ssperr_params,
        )
    elif mode == "fix_ssperr_merging":
        params_bounded._replace(
            ssperr_params=params_b_fixed.ssperr_params,
            merging_params=params_b_fixed.merging_params,
        )
    elif mode != "fit_all":
        valid_modes = [
            "fit_all",
            "fit_diffstarpop",
            "fit_ssperr",
            "fit_merging",
            "fix_diffstarpop",
            "fix_ssperr",
            "fix_merging",
            "fix_diffstarpop_merging",
            "fix_diffstarpop_ssperr",
            "fix_ssperr_merging",
        ]
        raise ValueError(
            "Invalid mode. Choose from the following:"
            + "\n".join(f" - {m}" for m in valid_modes)
        )

    for zi in range(N_z_bins):
        lc_data = lc_data_all[zi]
        n_gal = n_gal_all[zi]

        phot_info, phot_randoms, merging_randoms = gpkm._mc_phot_kern_merging(
            sed_key,
            lc_data.z_obs,
            lc_data.t_obs,
            lc_data.mah_params,
            lc_data.ssp_data,
            lc_data.precomputed_ssp_mag_table,
            lc_data.z_phot_table,
            lc_data.wave_eff_table,
            params_bounded.diffstarpop_params,
            params_bounded.mzr_params,
            params_bounded.spspop_params,
            params_bounded.scatter_params,
            params_bounded.ssperr_params,
            params_bounded.merging_params,
            DEFAULT_COSMOLOGY,
            FB,
            lc_data.logmp_infall,
            lc_data.logmhost_infall,
            lc_data.t_infall,
            lc_data.is_central,
            lc_data.sat_weight,
            lc_data.halo_indx,
            mc_merge=0,
        )

        for mi in range(n_mag):
            if mi <= n_mag - 2:
                mi2 = mi + 1
            else:
                mi2 = 0
            if mi >= 1:
                mi1 = mi - 1
            else:
                mi1 = n_mag - 1

            M_c_min_ = M_c_min_all[kk]
            M_c_max_ = M_c_max_all[kk]

            ndsig_M_c_pred_ = ndsig_M_c_pred_all[kk]

            M_c_data_ = M_c_data_all[kk]

            phot_info_obs_mags_wIGM = igm_attenuation(
                phot_info.obs_mags_weighted,
                lc_data.z_obs,
            )

            phot_info_data = jnp.vstack(
                (
                    phot_info_obs_mags_wIGM[:, mi],
                    phot_info_obs_mags_wIGM[:, i_i],
                    phot_info_obs_mags_wIGM[:, mi2] - phot_info_obs_mags_wIGM[:, mi],
                    phot_info_obs_mags_wIGM[:, mi1] - phot_info_obs_mags_wIGM[:, mi],
                )
            ).T

            phot_info_data_dropout = assign_dropout_values_CW_model(
                phot_info_data,
                mi,
                mi2,
                mi1,
                phot_info,
            )

            mask = phot_info_data_dropout[:, i_i] > 27.0
            phot_info_data_dropout = jnp.where(
                mask[:, None],
                999.0,
                phot_info_data_dropout,
            )

            M_c_pred_ = signdhist_lomem.nnsig_ndhist_weighted(
                phot_info_data_dropout,
                ndsig_M_c_pred_,
                n_gal,
                M_c_min_,
                M_c_max_,
            )

            if loss_type == "log_mse_loss":
                eps = 1e-1  # to avoid log(0) issues
                sum_1 = sum_1 + log_mse_loss(M_c_data_, M_c_pred_, eps)
            elif loss_type == "mse_loss":
                sum_1 = sum_1 + mse_loss(M_c_data_, M_c_pred_, eps)
            elif loss_type == "ln_poisson_loss":
                eps = 1e-3  # to avoid log(0) issues
                sum_1 = sum_1 + ln_poisson_loss(M_c_data_, M_c_pred_, eps)
            else:
                valid_modes = [
                    "log_mse_loss",
                    "mse_loss",
                    "ln_poisson_loss",
                ]
                raise ValueError(
                    "Invalid loss_type. Choose from the following:"
                    + "\n".join(f" - {m}" for m in valid_modes)
                )

            kk = kk + 1

    return sum_1 / norm
