from . import load_ssp_templates
from . import load_transmission_curve


def sdss_filters(sdss_filters_to_use):
    filter_options = [
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
    ]
    sdss_mag_options = [
        "modelMag_u",
        "modelMag_g",
        "modelMag_r",
        "modelMag_i",
        "modelMag_z",
    ]

    tcurves = []
    sdss_mag_colnames = []
    for bn_pat in sdss_filters_to_use:
        tcurve = load_transmission_curve(bn_pat=bn_pat + "*")
        tcurves.append(tcurve)

        switch = 0
        for j in range(len(filter_options)):
            if bn_pat == filter_options[j]:
                sdss_mag_colnames.append(sdss_mag_options[j])
                switch = 1

        if switch == 0:
            print("ERROR: incorrect sdss filter option.")
            print("Available options are:")
            for option in filter_options:
                print(option)

    ssp_data = load_ssp_templates()

    return tcurves, ssp_data, sdss_mag_colnames
