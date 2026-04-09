from . import sdss_loader as sdl
from astropy.table import Table

def get_clean_sdss_data():
    sdss = sdl.load_sdss_wrapper_get_mags_arr()
    return Table(sdss, names=('sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z'))

