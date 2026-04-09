import os
from . import io_utils as iou
from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm

BNPAT_PARAM_COLLECTION = "diffsky_{0}_param_collection.hdf5"
from collections import namedtuple


def load_param_collection(drn_mock, mock_version_name):
    """"""
    bn = BNPAT_PARAM_COLLECTION.format(mock_version_name)
    fn = os.path.join(drn_mock, bn)
    fn = os.path.expanduser(fn)
    flat_diffsky_params = iou.load_namedtuple_from_hdf5(fn)
    param_collection = dpwm.get_param_collection_from_flat_array(flat_diffsky_params)
    return param_collection


def write_param_collection(drn_mock, mock_version_name, param_collection):
    """"""
    bn = BNPAT_PARAM_COLLECTION.format(mock_version_name)
    fn_out = os.path.join(drn_mock, bn)
    fn_out = os.path.expanduser(fn_out)
    os.makedirs(os.path.dirname(fn_out), exist_ok=True)
    flat_diffsky_params = dpwm.unroll_param_collection_into_flat_array(
        *param_collection
    )
    DiffskyParams = namedtuple("DiffskyParams", dpwm.get_flat_param_names())
    flat_diffsky_params = DiffskyParams(*flat_diffsky_params)

    iou.write_namedtuple_to_hdf5(flat_diffsky_params, fn_out)
