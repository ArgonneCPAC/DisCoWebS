import numpy as np
import jax.numpy as jnp
from jax import random as jran
from diffsky.experimental import lightcone_generators as lcg
from diffsky.experimental.mc_phot import mc_lc_phot_merging
from dsps.cosmology import DEFAULT_COSMOLOGY
from diffstar.defaults import FB
from diffsky import signdhist
from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm
from jax import jit as jjit
from functools import partial
from .likelihood_kernel import (
    mse_loss,
    log_mse_loss,
    ln_poisson_loss,
)


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
        lgmp_min, lgmp_max = 10.7 + zi * 0.5 / N_z_bins, 14.8

        z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

        halo_lc_data = (num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area)
        phot_data = (ssp_data, tcurves, z_phot_table)

        ran_key, lc_halo_key = jran.split(ran_key, 2)
        args = (lc_halo_key, *halo_lc_data, *phot_data)

        lc_data = lcg.weighted_lc_photdata(*args)
        lc_data_all.append(lc_data)

        n_gal = np.ones_like(lc_data.nhalos_host)
        n_gal[:num_halos] = lc_data.nhalos[:num_halos]
        n_gal[num_halos:] = lc_data.nhalos[num_halos:] * lc_data.nhalos_host[num_halos:]
        n_gal_all.append(n_gal)

        ran_key, sed_key = jran.split(ran_key, 2)

        phot_info = mc_lc_phot_merging(
            sed_key,
            lc_data,
            diffstarpop_params=params_b.diffstarpop_params,
            mzr_params=params_b.mzr_params,
            spspop_params=params_b.spspop_params,
            scatter_params=params_b.scatter_params,
            ssperr_params=params_b.ssperr_params,
            merging_params=params_b.merging_params,
            cosmo_params=DEFAULT_COSMOLOGY,
            fb=FB,
            skip_param_check=True,
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

            bins = [N_mag_bins, N_mag_bins, N_color_bins, N_color_bins]

            Hist_nD1, bins = np.histogramdd(
                np.vstack(
                    (
                        cosmos_subset_cut[cosmos_mag_colnames[mi]],
                        cosmos_subset_cut["HSC_i_MAG"],
                        cosmos_subset_cut[cosmos_mag_colnames[mi2]]
                        - cosmos_subset_cut[cosmos_mag_colnames[mi]],
                        cosmos_subset_cut[cosmos_mag_colnames[mi1]]
                        - cosmos_subset_cut[cosmos_mag_colnames[mi]],
                    )
                ).T,
                bins=bins,
            )

            Hist_nD2, bins = np.histogramdd(
                np.vstack(
                    (
                        phot_info["obs_mags"][:, mi],
                        phot_info["obs_mags"][:, i_i],
                        phot_info["obs_mags"][:, mi2] - phot_info["obs_mags"][:, mi],
                        phot_info["obs_mags"][:, mi1] - phot_info["obs_mags"][:, mi],
                    )
                ).T,
                bins=bins,
            )

            non_zero_indices = np.where((Hist_nD1 >= 1.0) & (Hist_nD2 >= 1.0))
            switch = 1
            for i in range(len(non_zero_indices[0])):
                i0 = non_zero_indices[0][i]
                i1 = non_zero_indices[1][i]
                i2 = non_zero_indices[2][i]
                i3 = non_zero_indices[3][i]
                if switch == 1:
                    _min_0 = np.array(
                        [bins[0][i0], bins[1][i1], bins[2][i2], bins[3][i3]]
                    )
                    _max_0 = np.array(
                        [
                            bins[0][i0 + 1],
                            bins[1][i1 + 1],
                            bins[2][i2 + 1],
                            bins[3][i3 + 1],
                        ]
                    )
                    switch = 0
                else:
                    _min_j = np.array(
                        [bins[0][i0], bins[1][i1], bins[2][i2], bins[3][i3]]
                    )
                    _max_j = np.array(
                        [
                            bins[0][i0 + 1],
                            bins[1][i1 + 1],
                            bins[2][i2 + 1],
                            bins[3][i3 + 1],
                        ]
                    )
                    _min_0 = np.vstack([_min_0, _min_j])
                    _max_0 = np.vstack([_max_0, _max_j])

            M_c_min_ = jnp.array(_min_0)
            M_c_max_ = jnp.array(_max_0)

            ndsig_M_c_data_ = jnp.ones_like(
                np.vstack(
                    (
                        cosmos_subset[cosmos_mag_colnames[mi]],
                        cosmos_subset["HSC_i_MAG"],
                        cosmos_subset[cosmos_mag_colnames[mi2]]
                        - cosmos_subset[cosmos_mag_colnames[mi]],
                        cosmos_subset[cosmos_mag_colnames[mi1]]
                        - cosmos_subset[cosmos_mag_colnames[mi]],
                    )
                ).T
            )
            for j in range(len(ndsig_M_c_data_[0, :])):
                ndsig_M_c_data_ = ndsig_M_c_data_.at[:, j].set(
                    (bins[j][1] - bins[j][0]) * ndsig_by_dbin
                )

            ndsig_M_c_pred_ = jnp.ones_like(
                np.vstack(
                    (
                        phot_info["obs_mags"][:, mi],
                        phot_info["obs_mags"][:, i_i],
                        phot_info["obs_mags"][:, mi2] - phot_info["obs_mags"][:, mi],
                        phot_info["obs_mags"][:, mi1] - phot_info["obs_mags"][:, mi],
                    )
                ).T
            )
            for j in range(len(ndsig_M_c_pred_[0, :])):
                ndsig_M_c_pred_ = ndsig_M_c_pred_.at[:, j].set(
                    (bins[j][1] - bins[j][0]) * ndsig_by_dbin
                )

            M_c_data_ = signdhist.nnsig_ndhist(
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
        lgmp_min, lgmp_max = 10.9 + zi * 0.5 / N_z_bins, 15.0

        z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

        halo_lc_data = (num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area)
        phot_data = (ssp_data, tcurves, z_phot_table)

        ran_key, lc_halo_key = jran.split(ran_key, 2)
        args = (lc_halo_key, *halo_lc_data, *phot_data)

        lc_data = lcg.weighted_lc_photdata(*args)
        lc_data_all.append(lc_data)

        n_gal = np.ones_like(lc_data.nhalos_host)
        n_gal[:num_halos] = lc_data.nhalos[:num_halos]
        n_gal[num_halos:] = lc_data.nhalos[num_halos:] * lc_data.nhalos_host[num_halos:]
        n_gal_all.append(n_gal)

        ran_key, sed_key = jran.split(ran_key, 2)

        phot_info = mc_lc_phot_merging(
            sed_key,
            lc_data,
            diffstarpop_params=params_b.diffstarpop_params,
            mzr_params=params_b.mzr_params,
            spspop_params=params_b.spspop_params,
            scatter_params=params_b.scatter_params,
            ssperr_params=params_b.ssperr_params,
            merging_params=params_b.merging_params,
            cosmo_params=DEFAULT_COSMOLOGY,
            fb=FB,
            skip_param_check=True,
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

            bins = [N_mag_bins, N_mag_bins, N_color_bins, N_color_bins]

            Hist_nD1, bins = np.histogramdd(
                np.vstack(
                    (
                        sdss_subset_cut[sdss_mag_colnames[mi]],
                        sdss_subset_cut["modelMag_r"],
                        sdss_subset_cut[sdss_mag_colnames[mi2]]
                        - sdss_subset_cut[sdss_mag_colnames[mi]],
                        sdss_subset_cut[sdss_mag_colnames[mi1]]
                        - sdss_subset_cut[sdss_mag_colnames[mi]],
                    )
                ).T,
                bins=bins,
            )

            Hist_nD2, bins = np.histogramdd(
                np.vstack(
                    (
                        phot_info["obs_mags"][:, mi],
                        phot_info["obs_mags"][:, i_i],
                        phot_info["obs_mags"][:, mi2] - phot_info["obs_mags"][:, mi],
                        phot_info["obs_mags"][:, mi1] - phot_info["obs_mags"][:, mi],
                    )
                ).T,
                bins=bins,
            )

            non_zero_indices = np.where((Hist_nD1 >= 1.0) & (Hist_nD2 >= 1.0))
            switch = 1
            for i in range(len(non_zero_indices[0])):
                i0 = non_zero_indices[0][i]
                i1 = non_zero_indices[1][i]
                i2 = non_zero_indices[2][i]
                i3 = non_zero_indices[3][i]
                if switch == 1:
                    _min_0 = np.array(
                        [bins[0][i0], bins[1][i1], bins[2][i2], bins[3][i3]]
                    )
                    _max_0 = np.array(
                        [
                            bins[0][i0 + 1],
                            bins[1][i1 + 1],
                            bins[2][i2 + 1],
                            bins[3][i3 + 1],
                        ]
                    )
                    switch = 0
                else:
                    _min_j = np.array(
                        [bins[0][i0], bins[1][i1], bins[2][i2], bins[3][i3]]
                    )
                    _max_j = np.array(
                        [
                            bins[0][i0 + 1],
                            bins[1][i1 + 1],
                            bins[2][i2 + 1],
                            bins[3][i3 + 1],
                        ]
                    )
                    _min_0 = np.vstack([_min_0, _min_j])
                    _max_0 = np.vstack([_max_0, _max_j])

            M_c_min_ = jnp.array(_min_0)
            M_c_max_ = jnp.array(_max_0)

            ndsig_M_c_data_ = jnp.ones_like(
                np.vstack(
                    (
                        sdss_subset[sdss_mag_colnames[mi]],
                        sdss_subset["modelMag_r"],
                        sdss_subset[sdss_mag_colnames[mi2]]
                        - sdss_subset[sdss_mag_colnames[mi]],
                        sdss_subset[sdss_mag_colnames[mi1]]
                        - sdss_subset[sdss_mag_colnames[mi]],
                    )
                ).T
            )
            for j in range(len(ndsig_M_c_data_[0, :])):
                ndsig_M_c_data_ = ndsig_M_c_data_.at[:, j].set(
                    (bins[j][1] - bins[j][0]) * ndsig_by_dbin
                )

            ndsig_M_c_pred_ = jnp.ones_like(
                np.vstack(
                    (
                        phot_info["obs_mags"][:, mi],
                        phot_info["obs_mags"][:, i_i],
                        phot_info["obs_mags"][:, mi2] - phot_info["obs_mags"][:, mi],
                        phot_info["obs_mags"][:, mi1] - phot_info["obs_mags"][:, mi],
                    )
                ).T
            )
            for j in range(len(ndsig_M_c_pred_[0, :])):
                ndsig_M_c_pred_ = ndsig_M_c_pred_.at[:, j].set(
                    (bins[j][1] - bins[j][0]) * ndsig_by_dbin
                )

            M_c_data_ = signdhist.nnsig_ndhist(
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

        phot_info = mc_lc_phot_merging(
            sed_key,
            lc_data,
            diffstarpop_params=params_bounded.diffstarpop_params,
            mzr_params=params_bounded.mzr_params,
            spspop_params=params_bounded.spspop_params,
            scatter_params=params_bounded.scatter_params,
            ssperr_params=params_bounded.ssperr_params,
            merging_params=params_bounded.merging_params,
            cosmo_params=DEFAULT_COSMOLOGY,
            fb=FB,
            skip_param_check=True,
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

            M_c_pred_ = signdhist.nnsig_ndhist_weighted(
                jnp.vstack(
                    (
                        phot_info["obs_mags"][:, mi],
                        phot_info["obs_mags"][:, i_i],
                        phot_info["obs_mags"][:, mi2] - phot_info["obs_mags"][:, mi],
                        phot_info["obs_mags"][:, mi1] - phot_info["obs_mags"][:, mi],
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
        # "n_gal_all_combined",
        # "lc_data_all_combined",
        # "M_c_data_all_combined",
        # "ndsig_M_c_pred_all_combined",
        # "M_c_min_all_combined",
        # "M_c_max_all_combined",
        "N_z_bins",
        "i_i_combined",
        "n_mag_combined",
        "mode",
        "loss_type",
        "norm",
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
