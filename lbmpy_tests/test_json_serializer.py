"""
Test the lbmpy-specific JSON encoder and serializer as used in the Database class.
"""

import tempfile
import pystencils as ps

from lbmpy import Stencil, Method, ForceModel
from lbmpy.advanced_streaming import Timestep
from lbmpy.creationfunctions import LBMConfig, LBMOptimisation, LBStencil
from lbmpy.fieldaccess import StreamPullTwoFieldsAccessor

from pystencils.runhelper import Database
from lbmpy.db import LbmpyJsonSerializer


def test_json_serializer():

    stencil = LBStencil(Stencil.D3Q27)
    q = stencil.Q
    pdfs, pdfs_tmp = ps.fields(f"pdfs({q}), pdfs_tmp({q}): double[3D]", layout='fzyx')
    density = ps.fields(f"rho: double[3D]", layout='fzyx')

    from lbmpy.non_newtonian_models import CassonsParameters
    cassons_params = CassonsParameters(0.2)

    # create dummy lbmpy config
    lbm_config = LBMConfig(stencil=LBStencil(Stencil.D3Q27), method=Method.CUMULANT, force_model=ForceModel.GUO,
                           compressible=True, relaxation_rate=1.999, smagorinsky=True, galilean_correction=True,
                           cassons=cassons_params, density_input=density, kernel_type=StreamPullTwoFieldsAccessor,
                           timestep=Timestep.BOTH)

    lbm_optimization = LBMOptimisation(cse_pdfs=False, cse_global=False, builtin_periodicity=(True, False, False),
                                       symbolic_field=pdfs, symbolic_temporary_field=pdfs_tmp)

    # create dummy database
    temp_dir = tempfile.TemporaryDirectory()
    db = Database(file=temp_dir.name, serializer_info=('lbmpy_serializer', LbmpyJsonSerializer))

    db.save(params={'lbm_config': lbm_config, 'lbm_optimization': lbm_optimization}, result={'test': 'dummy'})
