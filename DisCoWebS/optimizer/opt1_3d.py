from DisCoWebS import config
from jax import random as jran
from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm
from jax import jit as jjit
import optax
from jax import value_and_grad, clear_caches
from functools import partial
from ..data_loader.param_collection import write_param_collection

from ..likelihood.likelihood_3d import (
    m_i_c_loss,
    bin_cosmos_data_m_i_c,
    bin_sdss_data_m_i_c,
    bin_cosmos_web_data_m_i_c,
    m_i_c_loss_multi_data,
)


def initialize_parameters(init_mode="initial", params_u=None, learn_rate=1e-1):
    if init_mode == "initial":
        default_u = dpwm.get_u_param_collection_from_param_collection(
            dpwm.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
            dpwm.DEFAULT_PARAM_COLLECTION.mzr_params,
            dpwm.DEFAULT_PARAM_COLLECTION.spspop_params,
            dpwm.DEFAULT_PARAM_COLLECTION.scatter_params,
            dpwm.DEFAULT_PARAM_COLLECTION.ssperr_params,
            dpwm.DEFAULT_PARAM_COLLECTION.merging_params,
        )
        params_u = default_u
    elif init_mode == "continue":
        if params_u is None:
            raise ValueError("For 'continue' init_mode, must define params_u")
        params_u = params_u
    else:
        raise ValueError("Invalid init_mode. Choose 'initial' or 'continue'.")
    learning_rate = learn_rate
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params_u)
    return params_u, opt_state, optimizer


@partial(
    jjit,
    static_argnames=[
        "optimizer",
        "i_i",
        "n_mag",
        "N_z_bins",
        "loss_function",
        "mode",
        "loss_type",
        "norm",
    ],
)
def train_step(
    params_u,
    opt_state,
    optimizer,
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
    loss_function,
    mode="fit_all",
    loss_type="log_mse_loss",
    norm=1.0,
):
    """
    Perform a single optimization step
    (forward pass, gradient calculation, and parameter update).
    """
    # Calculate loss and gradients

    i_i = int(i_i)
    n_mag = int(n_mag)
    N_z_bins = int(N_z_bins)
    mode = str(mode)
    loss_type = str(loss_type)
    norm = float(norm)

    loss_value, grads = value_and_grad(loss_function)(
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
        mode=mode,
        loss_type=loss_type,
        norm=norm,
    )

    # Apply optimizer updates
    updates, opt_state = optimizer.update(grads, opt_state, params_u)

    # Apply the updates to the parameters
    params_u = optax.apply_updates(params_u, updates)

    return params_u, opt_state, loss_value


@partial(
    jjit,
    static_argnames=[
        "optimizer",
        "loss_function",
        "mode",
        "loss_type",
        "norm",
        "N_z_bins",
        "i_i_combined",
        "n_mag_combined",
    ],
)
def train_step_multi_data(
    params_u,
    opt_state,
    optimizer,
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
    loss_function,
    mode="fit_all",
    loss_type="log_mse_loss",
    norm=1.0,
):
    """
    Perform a single optimization step
    (forward pass, gradient calculation, and parameter update).
    """
    # Calculate loss and gradients

    mode = str(mode)
    loss_type = str(loss_type)
    norm = float(norm)

    loss_value, grads = value_and_grad(loss_function)(
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
        params_b_fixed=params_b_fixed,
        mode=mode,
        loss_type=loss_type,
        norm=norm,
    )

    # Apply optimizer updates
    updates, opt_state = optimizer.update(grads, opt_state, params_u)

    # Apply the updates to the parameters
    params_u = optax.apply_updates(params_u, updates)

    return params_u, opt_state, loss_value


def one_loop_optimization(
    params_u,
    opt_state,
    optimizer,
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
    loss_function,
    num_steps=100,
    mode="fit_all",
    loss_type="log_mse_loss",
    norm=1.0,
):
    """
    Perform one loop of optimization
    and return the updated parameters and loss value.
    """

    i_i = int(i_i)
    n_mag = int(n_mag)
    N_z_bins = int(N_z_bins)
    mode = str(mode)
    loss_type = str(loss_type)
    norm = float(norm)

    for step in range(num_steps):
        params_u, opt_state, loss_value = train_step(
            params_u,
            opt_state,
            optimizer,
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
            loss_function,
            mode=mode,
            loss_type=loss_type,
            norm=norm,
        )
        if step == 0:
            min_loss = float(loss_value)

        if float(loss_value) < min_loss:
            min_loss = float(loss_value)
            params_u_bf = params_u

    return params_u_bf, min_loss


def one_loop_optimization_multi_data(
    params_u,
    opt_state,
    optimizer,
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
    loss_function,
    num_steps=100,
    mode="fit_all",
    loss_type="log_mse_loss",
    norm=1.0,
    data_to_use=("cosmos", "sdss"),
):
    """
    Perform one loop of optimization
    and return the updated parameters and loss value.
    """

    mode = str(mode)
    loss_type = str(loss_type)
    norm = float(norm)

    for step in range(num_steps):
        params_u, opt_state, loss_value = train_step_multi_data(
            params_u,
            opt_state,
            optimizer,
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
            loss_function,
            mode=mode,
            loss_type=loss_type,
            norm=norm,
        )
        if step == 0:
            min_loss = float(loss_value)

        if float(loss_value) < min_loss:
            min_loss = float(loss_value)
            params_u_bf = params_u

    return params_u_bf, min_loss


def multi_loop_optimization(
    params_u,
    opt_state,
    optimizer,
    data_mag_colnames,
    params_b_fixed,
    loss_function,
    data,
    i_i,
    n_mag,
    bin_data_m_i_c,
    N_z_bins,
    N_host_min,
    N_host_max,
    n_z_phot_table,
    ssp_data,
    tcurves,
    ran_key,
    N_mag_bins,
    N_color_bins,
    ndsig_by_dbin,
    num_loops=10,
    num_steps_per_loop=100,
    mode="fit_all",
    loss_type="log_mse_loss",
    norm=1.0,
):
    """
    Perform multiple loops of optimization
    and return the best parameters and loss value.
    """

    i_i = int(i_i)
    n_mag = int(n_mag)
    N_z_bins = int(N_z_bins)
    mode = str(mode)
    loss_type = str(loss_type)
    norm = float(norm)

    best_params_u = params_u
    best_loss = float("inf")

    params_bf_bounded = dpwm.get_param_collection_from_u_param_collection(
        best_params_u.diffstarpop_u_params,
        best_params_u.mzr_u_params,
        best_params_u.spspop_u_params,
        best_params_u.scatter_u_params,
        best_params_u.ssperr_u_params,
        best_params_u.merging_u_params,
    )

    for loop in range(num_loops):
        (
            ran_key,
            lc_data_all,
            M_c_data_all,
            ndsig_M_c_pred_all,
            M_c_min_all,
            M_c_max_all,
            n_gal_all,
        ) = bin_data_m_i_c(
            data,
            data_mag_colnames,
            N_z_bins,
            N_host_min,
            N_host_max,
            params_bf_bounded,
            n_z_phot_table,
            ssp_data,
            tcurves,
            ran_key,
            N_mag_bins,
            N_color_bins,
            ndsig_by_dbin,
        )

        ran_key, sed_key = jran.split(ran_key, 2)

        params_u, opt_state, optimizer = initialize_parameters(
            init_mode="continue", params_u=best_params_u, learn_rate=1e-1
        )

        params_u, opt_state, loss_value = train_step(
            params_u,
            opt_state,
            optimizer,
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
            loss_function,
            mode=mode,
            loss_type=loss_type,
            norm=norm,
        )

        params_u, loss_value = one_loop_optimization(
            params_u,
            opt_state,
            optimizer,
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
            loss_function,
            num_steps=num_steps_per_loop,
            mode=mode,
            loss_type=loss_type,
            norm=norm,
        )

        if loss_value < best_loss:
            best_loss = loss_value
            best_params_u = params_u

        print(f"Loop {loop + 1}/{num_loops}, Loss: {loss_value:.4f}")

        params_bf_bounded = dpwm.get_param_collection_from_u_param_collection(
            best_params_u.diffstarpop_u_params,
            best_params_u.mzr_u_params,
            best_params_u.spspop_u_params,
            best_params_u.scatter_u_params,
            best_params_u.ssperr_u_params,
            best_params_u.merging_u_params,
        )

    return params_bf_bounded, best_loss


def multi_loop_optimization_multi_data(
    params_u,
    opt_state,
    optimizer,
    data_mag_colnames,
    params_b_fixed,
    loss_function,
    data,
    i_i,
    n_mag,
    N_z_bins,
    N_host_min,
    N_host_max,
    n_z_phot_table,
    ssp_data,
    tcurves,
    ran_key,
    N_mag_bins,
    N_color_bins,
    ndsig_by_dbin,
    num_loops=10,
    num_steps_per_loop=100,
    mode="fit_all",
    loss_type="log_mse_loss",
    norm=1.0,
    data_to_use=("cosmos", "sdss"),
):
    """
    Perform multiple loops of optimization for multiple datasets
    and return the best parameters and loss value.
    """

    best_params_u = params_u
    best_loss = float("inf")

    params_bf_bounded = dpwm.get_param_collection_from_u_param_collection(
        best_params_u.diffstarpop_u_params,
        best_params_u.mzr_u_params,
        best_params_u.spspop_u_params,
        best_params_u.scatter_u_params,
        best_params_u.ssperr_u_params,
        best_params_u.merging_u_params,
    )

    for loop in range(num_loops):
        lc_data_all_combined = []
        M_c_data_all_combined = []
        M_c_max_all_combined = []
        ndsig_M_c_pred_all_combined = []
        M_c_min_all_combined = []
        n_gal_all_combined = []
        n_mag_combined = []
        i_i_combined = []

        for di in range(len(data_to_use)):
            if data_to_use[di] == "cosmos":
                bin_data_m_i_c = bin_cosmos_data_m_i_c
                i_i = data_mag_colnames[di].index("HSC_i_MAG")
            elif data_to_use[di] == "sdss":
                bin_data_m_i_c = bin_sdss_data_m_i_c
                i_i = data_mag_colnames[di].index("modelMag_r")
            elif data_to_use[di] == "cosmos_web":
                bin_data_m_i_c = bin_cosmos_web_data_m_i_c
                i_i = data_mag_colnames[di].index("mag_model_f444w")
            else:
                raise ValueError(
                    "data_to_use: Choose 'cosmos', 'sdss', or 'cosmos_web'."
                )

            (
                ran_key,
                lc_data_all,
                M_c_data_all,
                ndsig_M_c_pred_all,
                M_c_min_all,
                M_c_max_all,
                n_gal_all,
            ) = bin_data_m_i_c(
                data[di],
                data_mag_colnames[di],
                N_z_bins[di],
                N_host_min[di],
                N_host_max[di],
                params_bf_bounded,
                n_z_phot_table[di],
                ssp_data[di],
                tcurves[di],
                ran_key,
                N_mag_bins[di],
                N_color_bins[di],
                ndsig_by_dbin,
            )

            lc_data_all_combined.append(lc_data_all)
            M_c_data_all_combined.append(M_c_data_all)
            ndsig_M_c_pred_all_combined.append(ndsig_M_c_pred_all)
            M_c_min_all_combined.append(M_c_min_all)
            M_c_max_all_combined.append(M_c_max_all)
            n_gal_all_combined.append(n_gal_all)
            i_i_combined.append(i_i)
            n_mag = len(data_mag_colnames[di])
            n_mag_combined.append(n_mag)

        lc_data_all_combined = tuple(lc_data_all_combined)
        M_c_data_all_combined = tuple(M_c_data_all_combined)
        ndsig_M_c_pred_all_combined = tuple(ndsig_M_c_pred_all_combined)
        M_c_min_all_combined = tuple(M_c_min_all_combined)
        M_c_max_all_combined = tuple(M_c_max_all_combined)
        n_gal_all_combined = tuple(n_gal_all_combined)
        i_i_combined = tuple(i_i_combined)
        n_mag_combined = tuple(n_mag_combined)

        ran_key, sed_key = jran.split(ran_key, 2)

        params_u, opt_state, optimizer = initialize_parameters(
            init_mode="continue", params_u=best_params_u, learn_rate=1e-1
        )

        params_u, opt_state, loss_value = train_step_multi_data(
            params_u,
            opt_state,
            optimizer,
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
            loss_function,
            mode=mode,
            loss_type=loss_type,
            norm=norm,
        )

        params_u, loss_value = one_loop_optimization_multi_data(
            params_u,
            opt_state,
            optimizer,
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
            loss_function,
            num_steps=num_steps_per_loop,
            mode=mode,
            loss_type=loss_type,
            norm=norm,
            data_to_use=data_to_use,
        )

        if loss_value < best_loss:
            best_loss = loss_value
            best_params_u = params_u

        print(f"Loop {loop + 1}/{num_loops}, Loss: {loss_value:.4f}")

        params_bf_bounded = dpwm.get_param_collection_from_u_param_collection(
            best_params_u.diffstarpop_u_params,
            best_params_u.mzr_u_params,
            best_params_u.spspop_u_params,
            best_params_u.scatter_u_params,
            best_params_u.ssperr_u_params,
            best_params_u.merging_u_params,
        )

        clear_caches()

    return params_bf_bounded, best_loss


def run_optimization_1data(
    data,
    data_mag_colnames,
    N_z_bins,
    N_host_min,
    N_host_max,
    n_z_phot_table,
    ssp_data,
    tcurves,
    N_mag_bins,
    N_color_bins,
    ndsig_by_dbin,
    num_loops=10,
    num_steps_per_loop=100,
    modes=["fix_diffstarpop_merging", "fix_diffstarpop", "fit_all"],
    data_to_use="cosmos",
    loss_type="log_mse_loss",
    init_params_bounded=None,
):
    """
    Run the optimization process.
    """
    ran_key = jran.key(0)

    params_u, opt_state, optimizer = initialize_parameters(
        init_mode="initial", learn_rate=1e-1
    )

    if data_to_use == "cosmos":
        bin_data_m_i_c = bin_cosmos_data_m_i_c
        i_i = data_mag_colnames.index("HSC_i_MAG")
    elif data_to_use == "sdss":
        bin_data_m_i_c = bin_sdss_data_m_i_c
        i_i = data_mag_colnames.index("modelMag_r")
    elif data_to_use == "cosmos_web":
        bin_data_m_i_c = bin_cosmos_web_data_m_i_c
        i_i = data_mag_colnames.index("mag_model_f444w")
    else:
        raise ValueError(
            "Invalid data_to_use. Choose 'cosmos', 'sdss', or 'cosmos_web'."
        )

    if init_params_bounded is not None:
        params_bf_bounded = init_params_bounded
    else:
        params_bf_bounded = dpwm.DEFAULT_PARAM_COLLECTION

    (
        ran_key,
        lc_data_all,
        M_c_data_all,
        ndsig_M_c_pred_all,
        M_c_min_all,
        M_c_max_all,
        n_gal_all,
    ) = bin_data_m_i_c(
        data,
        data_mag_colnames,
        N_z_bins,
        N_host_min,
        N_host_max,
        params_bf_bounded,
        n_z_phot_table,
        ssp_data,
        tcurves,
        ran_key,
        N_mag_bins,
        N_color_bins,
        ndsig_by_dbin,
    )

    n_mag = len(data_mag_colnames)

    ran_key, sed_key = jran.split(ran_key, 2)

    init_loss = m_i_c_loss(
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
        params_b_fixed=params_bf_bounded,
        mode="fit_all",
        loss_type=loss_type,
        norm=1.0,
    )

    norm = init_loss * 1.0
    norm = float(norm)

    if loss_type == "ln_poisson_loss":
        print(f"Initial loss: {init_loss:.4f}")
    else:
        print("Initial Loss:  1.00 ")

    for opt_loop in range(len(modes)):
        mode = modes[opt_loop]
        print(f"Optimization {opt_loop + 1}/{len(modes)}, mode: {mode}")

        params_b_fixed = params_bf_bounded

        params_bf_bounded, best_loss = multi_loop_optimization(
            params_u,
            opt_state,
            optimizer,
            data_mag_colnames,
            params_b_fixed,
            m_i_c_loss,
            data,
            i_i,
            n_mag,
            bin_data_m_i_c,
            N_z_bins,
            N_host_min,
            N_host_max,
            n_z_phot_table,
            ssp_data,
            tcurves,
            ran_key,
            N_mag_bins,
            N_color_bins,
            ndsig_by_dbin,
            num_loops=num_loops,
            num_steps_per_loop=num_steps_per_loop,
            mode=mode,
            loss_type=loss_type,
            norm=norm,
        )

        params_u = dpwm.get_u_param_collection_from_param_collection(
            params_bf_bounded.diffstarpop_params,
            params_bf_bounded.mzr_params,
            params_bf_bounded.spspop_params,
            params_bf_bounded.scatter_params,
            params_bf_bounded.ssperr_params,
            params_bf_bounded.merging_params,
        )

    return params_bf_bounded, best_loss


def run_optimization_multi_data(
    data,
    data_mag_colnames,
    N_z_bins,
    N_host_min,
    N_host_max,
    n_z_phot_table,
    ssp_data,
    tcurves,
    N_mag_bins,
    N_color_bins,
    ndsig_by_dbin,
    num_loops=10,
    num_steps_per_loop=100,
    modes=("fix_diffstarpop_merging", "fix_diffstarpop", "fit_all"),
    data_to_use=("cosmos", "sdss"),
    loss_type="log_mse_loss",
    init_params_bounded=None,
):
    """
    Run the optimization process.
    """
    ran_key = jran.key(0)

    params_u, opt_state, optimizer = initialize_parameters(
        init_mode="initial", learn_rate=1e-1
    )

    (
        N_host_min,
        N_host_max,
        n_z_phot_table,
        ssp_data,
        tcurves,
        N_mag_bins,
        N_color_bins,
    ) = check_input_types_in_multidata(
        data_to_use,
        data,
        data_mag_colnames,
        N_z_bins,
        N_host_min,
        N_host_max,
        n_z_phot_table,
        ssp_data,
        tcurves,
        N_mag_bins,
        N_color_bins,
    )

    if init_params_bounded is not None:
        params_bf_bounded = init_params_bounded
    else:
        params_bf_bounded = dpwm.DEFAULT_PARAM_COLLECTION

    lc_data_all_combined = []
    M_c_data_all_combined = []
    M_c_max_all_combined = []
    ndsig_M_c_pred_all_combined = []
    M_c_min_all_combined = []
    n_gal_all_combined = []
    n_mag_combined = []
    i_i_combined = []

    for di in range(len(data_to_use)):
        if data_to_use[di] == "cosmos":
            bin_data_m_i_c = bin_cosmos_data_m_i_c
            i_i = data_mag_colnames[di].index("HSC_i_MAG")
        elif data_to_use[di] == "sdss":
            bin_data_m_i_c = bin_sdss_data_m_i_c
            i_i = data_mag_colnames[di].index("modelMag_r")
        elif data_to_use[di] == "cosmos_web":
            bin_data_m_i_c = bin_cosmos_web_data_m_i_c
            i_i = data_mag_colnames[di].index("mag_model_f444w")
        else:
            raise ValueError(
                "Invalid data_to_use. Choose 'cosmos', 'sdss', or 'cosmos_web'."
            )

        (
            ran_key,
            lc_data_all,
            M_c_data_all,
            ndsig_M_c_pred_all,
            M_c_min_all,
            M_c_max_all,
            n_gal_all,
        ) = bin_data_m_i_c(
            data[di],
            data_mag_colnames[di],
            N_z_bins[di],
            N_host_min[di],
            N_host_max[di],
            params_bf_bounded,
            n_z_phot_table[di],
            ssp_data[di],
            tcurves[di],
            ran_key,
            N_mag_bins[di],
            N_color_bins[di],
            ndsig_by_dbin,
        )

        lc_data_all_combined.append(lc_data_all)
        M_c_data_all_combined.append(M_c_data_all)
        ndsig_M_c_pred_all_combined.append(ndsig_M_c_pred_all)
        M_c_min_all_combined.append(M_c_min_all)
        M_c_max_all_combined.append(M_c_max_all)
        n_gal_all_combined.append(n_gal_all)
        i_i_combined.append(i_i)
        n_mag = len(data_mag_colnames[di])
        n_mag_combined.append(n_mag)

    lc_data_all_combined = tuple(lc_data_all_combined)
    M_c_data_all_combined = tuple(M_c_data_all_combined)
    ndsig_M_c_pred_all_combined = tuple(ndsig_M_c_pred_all_combined)
    M_c_min_all_combined = tuple(M_c_min_all_combined)
    M_c_max_all_combined = tuple(M_c_max_all_combined)
    n_gal_all_combined = tuple(n_gal_all_combined)
    i_i_combined = tuple(i_i_combined)
    n_mag_combined = tuple(n_mag_combined)

    ran_key, sed_key = jran.split(ran_key, 2)

    init_loss = m_i_c_loss_multi_data(
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
        params_b_fixed=params_bf_bounded,
        mode="fit_all",
        loss_type=loss_type,
        norm=1.0,
        data_to_use=data_to_use,
    )

    norm = init_loss * 1.0
    norm = float(norm)

    if loss_type == "ln_poisson_loss":
        print(f"Initial loss: {init_loss:.4f}")
    else:
        print("Initial Loss:  1.00 ")

    for opt_loop in range(len(modes)):
        mode = modes[opt_loop]
        print(f"Optimization {opt_loop + 1}/{len(modes)}, mode: {mode}")

        params_b_fixed = params_bf_bounded

        params_bf_bounded, best_loss = multi_loop_optimization_multi_data(
            params_u,
            opt_state,
            optimizer,
            data_mag_colnames,
            params_b_fixed,
            m_i_c_loss_multi_data,
            data,
            i_i_combined,
            n_mag_combined,
            N_z_bins,
            N_host_min,
            N_host_max,
            n_z_phot_table,
            ssp_data,
            tcurves,
            ran_key,
            N_mag_bins,
            N_color_bins,
            ndsig_by_dbin,
            num_loops=num_loops,
            num_steps_per_loop=num_steps_per_loop,
            mode=mode,
            loss_type=loss_type,
            norm=norm,
            data_to_use=data_to_use,
        )

        params_u = dpwm.get_u_param_collection_from_param_collection(
            params_bf_bounded.diffstarpop_params,
            params_bf_bounded.mzr_params,
            params_bf_bounded.spspop_params,
            params_bf_bounded.scatter_params,
            params_bf_bounded.ssperr_params,
            params_bf_bounded.merging_params,
        )

        # Save the best-fit parameters:
        write_param_collection(
            drn_mock="~/calibrated_params",
            mock_version_name=config.mock_version_name + "_" + mode + "_3d",
            param_collection=params_bf_bounded,
        )

        clear_caches()

    return params_bf_bounded, best_loss


def raise_value_error_1data():
    raise ValueError(
        "If you want to fit just 1 type of data use 'run_optimization_1data'."
    )
    pass


def check_input_types_in_multidata(
    data_to_use,
    data,
    data_mag_colnames,
    N_z_bins,
    N_host_min,
    N_host_max,
    n_z_phot_table,
    ssp_data,
    tcurves,
    N_mag_bins,
    N_color_bins,
):
    if not isinstance(data_to_use, tuple):
        print("data_to_use should be a tuple of strings.")
        print("Example: data_to_use=('cosmos', 'sdss')")
        raise_value_error_1data()

    if not isinstance(data, tuple):
        print("data should be a tuple of dataframes")
        print("Each dataframe should correspond to a dataset in data_to_use")
        print("Example: data=(cosmos, sdss)")
        raise_value_error_1data()

    if not isinstance(data_mag_colnames, tuple):
        print("data_mag_colnames should be a tuple of lists of strings")
        print("Each inner list should correspond to a dataset in data_to_use")
        print("e.g.:data_mag_colnames=(cosmos_mag_colnames,sdss_mag_colnames)")
        raise_value_error_1data()

    if not isinstance(N_z_bins, tuple):
        print("N_z_bins should be a tuple of integers")
        print("Each integer should correspond to a dataset in data_to_use")
        print("Example: N_z_bins=(10, 10)")
        raise_value_error_1data()

    if not isinstance(N_host_min, tuple) or not isinstance(N_host_max, tuple):
        N_host_min = (N_host_min,) * len(data_to_use)
        N_host_max = (N_host_max,) * len(data_to_use)

    if not isinstance(n_z_phot_table, tuple):
        n_z_phot_table = (n_z_phot_table,) * len(data_to_use)

    if not isinstance(ssp_data, tuple):
        ssp_data = (ssp_data,) * len(data_to_use)

    if not isinstance(tcurves, tuple):
        print("tcurves should be a tuple of tcurves objects")
        print("Each tcurves object corresponds to a dataset in data_to_use")
        print("Example: tcurves=(tcurves_cosmos, tcurves_sdss)")
        raise_value_error_1data()

    if not isinstance(N_mag_bins, tuple):
        N_mag_bins = (N_mag_bins,) * len(data_to_use)

    if not isinstance(N_color_bins, tuple):
        N_color_bins = (N_color_bins,) * len(data_to_use)

    return (
        N_host_min,
        N_host_max,
        n_z_phot_table,
        ssp_data,
        tcurves,
        N_mag_bins,
        N_color_bins,
    )
