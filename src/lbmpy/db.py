import json
import six
import inspect

from pystencils.runhelper.db import PystencilsJsonEncoder
from pystencils.simp import SimplificationStrategy
from lbmpy import LBStencil, Method, CollisionSpace
from lbmpy.creationfunctions import LBMConfig, LBMOptimisation
from lbmpy.methods import CollisionSpaceInfo
from lbmpy.forcemodels import AbstractForceModel
from lbmpy.non_newtonian_models import CassonsParameters


class LbmpyJsonEncoder(PystencilsJsonEncoder):

    def default(self, obj):
        if isinstance(obj, (LBMConfig, LBMOptimisation, CollisionSpaceInfo, CassonsParameters)):
            return obj.__dict__
        if isinstance(obj, (LBStencil, Method, CollisionSpace)):
            return obj.name
        if isinstance(obj, AbstractForceModel):
            return obj.__class__.__name__
        if isinstance(obj, SimplificationStrategy):
            return obj.__str__()
        if inspect.isclass(obj):
            return obj.__name__
        return PystencilsJsonEncoder.default(self, obj)


class LbmpyJsonSerializer(object):

    @classmethod
    def serialize(cls, data):
        if six.PY3:
            if isinstance(data, bytes):
                return json.dumps(data.decode('utf-8'), cls=LbmpyJsonEncoder, ensure_ascii=False).encode('utf-8')
            else:
                return json.dumps(data, cls=LbmpyJsonEncoder, ensure_ascii=False).encode('utf-8')
        else:
            return json.dumps(data, cls=LbmpyJsonEncoder, ensure_ascii=False).encode('utf-8')

    @classmethod
    def deserialize(cls, data):
        if six.PY3:
            return json.loads(data.decode('utf-8'))
        else:
            return json.loads(data.decode('utf-8'))
