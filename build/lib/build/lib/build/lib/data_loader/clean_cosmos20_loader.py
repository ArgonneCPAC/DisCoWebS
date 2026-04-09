from . import cosmos20_loader as c20


def get_clean_cosmos20_data():
    cosmos_all = c20.load_cosmos20()
    cosmos = c20.apply_nan_cuts(cosmos_all)
    msk_is_complete = c20.get_is_complete_mask(cosmos)
    cosmos = cosmos[msk_is_complete]

    msk_is_not_hsc_outlier = c20.get_color_outlier_mask(cosmos, c20.HSC_MAG_NAMES)
    msk_is_not_uvista_outlier = c20.get_color_outlier_mask(cosmos, c20.UVISTA_MAG_NAMES)
    msk_is_not_uvista_outlier.mean(), msk_is_not_hsc_outlier.mean()
    cosmos = cosmos[msk_is_not_hsc_outlier & msk_is_not_uvista_outlier]

    return cosmos
