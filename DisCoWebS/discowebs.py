from DisCoWebS.data_loader import clean_cosmos20_loader as cc20
from DisCoWebS.data_loader import sdss_loader as sdl
from DisCoWebS.data_loader import cosmos_web_loader as cwl
from DisCoWebS.data_loader import load_ssp_templates
from DisCoWebS.data_loader import sdss_filter_select
from DisCoWebS.data_loader import cosmos_web_filter_select
from DisCoWebS.optimizer import opt1_4d
from DisCoWebS.data_loader.param_collection import (
    load_param_collection,
    write_param_collection,
)
from DisCoWebS import config
from DisCoWebS.modelling.igm import load_igm_attenuation_table

if __name__ == "__main__":
    # Load data:
    sdss = sdl.load_sdss_wrapper()
    cosmos_web = cwl.load_cosmos_web_without_MIRI(
        app_mag_f444w_cut=27.0, z_min=0.3, z_max=3.0
    )

    # Choose filters. Load transmission curves and SSP templates.
    sdss_filters_to_use = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    tcurves_sdss, ssp_data_sdss, sdss_mag_colnames = sdss_filter_select.sdss_filters(
        sdss_filters_to_use
    )

    # Choose filters. Load transmission curves and SSP templates.
    cosmos_web_filters_to_use = [
        "chft_u",
        "hsc_g",
        "hsc_r",
        "hsc_i",
        "hsc_z",
        "hsc_y",
        "nircam_f115w",
        "nircam_f150w",
        "nircam_f277w",
        "nircam_f444w",
    ]
    tcurves_cosmos_web, ssp_data_cosmos_web, cosmos_web_mag_colnames = (
        cosmos_web_filter_select.cosmos_web_filters(cosmos_web_filters_to_use)
    )
    config.cosmos_web_filters_to_use = cosmos_web_filters_to_use

    # Load IGM attenuation interpolation table:
    config.igm_attenuation_table = load_igm_attenuation_table()

    # Choose a set of initial parameters:
    init_params_bounded = load_param_collection(
        drn_mock="~/calibrated_params/", mock_version_name="discowebs1"
    )

    # Run optimization:
    params_bf_bounded, best_loss = opt1_4d.run_optimization_multi_data(
        data=(sdss, cosmos_web),
        data_mag_colnames=(sdss_mag_colnames, cosmos_web_mag_colnames),
        N_z_bins=(2, 4),
        N_host_min=(1200, 1200),
        N_host_max=(1200, 1200),
        n_z_phot_table=(10, 16),
        ssp_data=(ssp_data_sdss, ssp_data_cosmos_web),
        tcurves=(tcurves_sdss, tcurves_cosmos_web),
        N_mag_bins=(8, 10),
        N_color_bins=(10, 10),
        ndsig_by_dbin=0.5,
        num_loops=4,
        num_steps_per_loop=4,
        modes=("fix_diffstarpop_merging", "fix_diffstarpop", "fit_all"),
        data_to_use=("sdss", "cosmos_web"),
        loss_type="log_mse_loss",
        init_params_bounded=init_params_bounded,
    )

    # Save the best-fit parameters:
    write_param_collection(
        drn_mock="~/calibrated_params",
        mock_version_name="discowebs1",
        param_collection=params_bf_bounded,
    )
