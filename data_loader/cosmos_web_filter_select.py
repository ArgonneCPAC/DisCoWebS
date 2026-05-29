from . import load_ssp_templates
from . import load_transmission_curve


def cosmos_web_filters(cosmos_web_filters_to_use):
    filter_options = [
        "chft_u",
        "hsc_g",
        "hsc_r",
        "hsc_i",
        "hsc_z",
        "hsc_y",
        "hst_f814w",
        "nircam_f115w",
        "nircam_f150w",
        "nircam_f277w",
        "nircam_f444w",
    ]

    cosmos_web_mag_options = [
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
        # "mag_model_f770w",
    ]

    tcurves = []
    cosmos_web_mag_colnames = []
    for bn_pat in cosmos_web_filters_to_use:
        tcurve = load_transmission_curve(bn_pat=bn_pat + "*")
        tcurves.append(tcurve)

        switch = 0
        for j in range(len(filter_options)):
            if bn_pat == filter_options[j]:
                cosmos_web_mag_colnames.append(cosmos_web_mag_options[j])
                switch = 1

        if switch == 0:
            print("ERROR: incorrect cosmos filter option")
            print("Available options are: " + str(filter_options))

    ssp_data = load_ssp_templates()

    return tcurves, ssp_data, cosmos_web_mag_colnames
