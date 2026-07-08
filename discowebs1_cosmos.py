from DisCoWebS.data_loader import clean_cosmos20_loader as cc20
from DisCoWebS.data_loader import cosmos_filter_select
from DisCoWebS.optimizer import opt1_4d
from DisCoWebS.data_loader.param_collection import (
    load_param_collection,
    write_param_collection,
)
from DisCoWebS import config

mock_version_name = "discos1_Nh1000_10_50"
config.mock_version_name = mock_version_name

if __name__ == "__main__":
    # Load data:
    cosmos = cc20.get_clean_cosmos20_data()

    # Choose filters. Load transmission curves and SSP templates.
    cosmos_filters_to_use = [
        "g_HSC",
        "r_HSC",
        "i_HSC",
        "z_HSC",
        "y_HSC",
        "Y_uv",
        "J_uv",
        "H_uv",
        "K_uv",
    ]
    tcurves_cosmos, ssp_data_cosmos, cosmos_mag_colnames = (
        cosmos_filter_select.cosmos_filters(cosmos_filters_to_use)
    )
    """
    # Choose a set of initial parameters:
    init_params_bounded = load_param_collection(
        drn_mock="~/calibrated_params/", mock_version_name="discowebs1"
    )
    """

    # Run optimization:
    params_bf_bounded, best_loss = opt1_4d.run_optimization_1data(
        data=cosmos,
        data_mag_colnames=cosmos_mag_colnames,
        N_z_bins=10,
        N_host_min=1000,
        N_host_max=1000,
        n_z_phot_table=10,
        ssp_data=ssp_data_cosmos,
        tcurves=tcurves_cosmos,
        N_mag_bins=10,
        N_color_bins=10,
        ndsig_by_dbin=0.5,
        num_loops=10,
        num_steps_per_loop=50,
        modes=["fix_diffstarpop_merging", "fix_diffstarpop", "fit_all"],
        data_to_use="cosmos",
        loss_type="log_mse_loss",
        init_params_bounded=None,
    )

    # Save the best-fit parameters:
    write_param_collection(
        drn_mock="~/calibrated_params",
        mock_version_name=config.mock_version_name,
        param_collection=params_bf_bounded,
    )
