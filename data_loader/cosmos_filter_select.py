from . import load_ssp_templates
from . import load_transmission_curve


def cosmos_filters(cosmos_filters_to_use):
    filter_options = [
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

    cosmos_mag_options = [
        "HSC_g_MAG",
        "HSC_r_MAG",
        "HSC_i_MAG",
        "HSC_z_MAG",
        "HSC_y_MAG",
        "UVISTA_Y_MAG",
        "UVISTA_J_MAG",
        "UVISTA_H_MAG",
        "UVISTA_Ks_MAG",
    ]

    tcurves = []
    cosmos_mag_colnames = []
    for bn_pat in cosmos_filters_to_use:
        tcurve = load_transmission_curve(bn_pat=bn_pat + "*")
        tcurves.append(tcurve)

        switch = 0
        for j in range(len(filter_options)):
            if bn_pat == filter_options[j]:
                cosmos_mag_colnames.append(cosmos_mag_options[j])
                switch = 1

        if switch == 0:
            print("ERROR: incorrect cosmos filter option")

    ssp_data = load_ssp_templates()

    return tcurves, ssp_data, cosmos_mag_colnames
