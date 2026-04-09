from jax import random as jran
from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm
from jax import jit as jjit
import optax
from jax import value_and_grad
from functools import partial

from ..likelihood.likelihood_4d import m_i_c1_c2_loss
from ..likelihood.likelihood_4d import bin_cosmos_data_m_i_c1_c2
from ..likelihood.likelihood_4d import bin_sdss_data_m_i_c1_c2


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
            min_loss = loss_value

        if loss_value < min_loss:
            min_loss = loss_value
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
    bin_data_m_i_c1_c2,
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
        ) = bin_data_m_i_c1_c2(
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
):
    """
    Run the optimization process.
    """
    ran_key = jran.key(0)

    params_u, opt_state, optimizer = initialize_parameters(
        init_mode="initial", learn_rate=1e-1
    )

    if data_to_use == "cosmos":
        bin_data_m_i_c1_c2 = bin_cosmos_data_m_i_c1_c2
        i_i = data_mag_colnames.index("HSC_i_MAG")
    elif data_to_use == "sdss":
        bin_data_m_i_c1_c2 = bin_sdss_data_m_i_c1_c2
        i_i = data_mag_colnames.index("modelMag_r")
    else:
        raise ValueError("Invalid data_to_use. Choose 'cosmos' or 'sdss'.")

    params_bf_bounded = dpwm.DEFAULT_PARAM_COLLECTION
    (
        ran_key,
        lc_data_all,
        M_c_data_all,
        ndsig_M_c_pred_all,
        M_c_min_all,
        M_c_max_all,
        n_gal_all,
    ) = bin_data_m_i_c1_c2(
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

    init_loss = m_i_c1_c2_loss(
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
            m_i_c1_c2_loss,
            data,
            i_i,
            n_mag,
            bin_data_m_i_c1_c2,
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
