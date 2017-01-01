import sympy as sp



class MomentBasedLbmMethod(AbstractLbmMethod):
    def __init__(self, stencil, moments, relaxationRates):
        pass


class FullStencilCumulantMethod(AbstractLbmMethod):
    def __init__(self):
        pass


class FullStencilMomentMethod():

    def __init__(self, ):
        pass


# -------------------------------------------------------------------------------------------------------------------


class RelaxationScheme:

    def doRelaxation(self, equationCollection):
        pass

    def equilibriumInCollisionSpace(self):
        """Returns a map from collision space component to its equilibrium value"""
        pass

    @property
    def conservedQuantities(self, name):
        """Returns """
        pass

    @property
    def symbolicRelaxationRates(self):
        pass


#class LbmMethod:
#    """
#    - splitting Relaxation and lattice model is bad - since relaxation may depend on set of methods
#
#    Class that holds all information about a lattice Boltzmann collision scheme:
#        - discretization of velocity space i.e. stencil
#        - equations to transform into collision space and back
#        - relaxation scheme and equilibrium
#        - conserved quantities and equations to compute them
#
#    Interface to force models and boundary conditions:
#        - getMacroscopicQuantity('velocity')
#        - getMacroscopicQuantity('density')
#        - getMacroscopicQuantitySymbol('velocity')
#
#    General Idea:
#        - separate collision space transformation (e.g. raw moments, central moments and cumulant)
#          from relaxation schemes (e.g. various orthogonalization methods)
#        - question: who defines macroscopic/conserved values
#
#    Open Questions: TODO
#        - forcemodel as member? or can it be supplied in getCollisionRule()
#          and getMacroscopicQuantity() -> probably a member is better
#
#    ForceModel Interface:
#        - pass velocity & compressibility instead of complete lbmMethod? not sure...
#        -
#    Boundary Interface:
#        - pass full lbm method to boundary function
#
#    """
#
#    def __init__(self, stencil, collisionSpaceTransformation, relaxationScheme,
#                 inverseCollisionSpaceTransformation=None):
#        """
#        Create a LbmMethod
#        :param stencil:
#                   sequence of directions, each direction is a tuple of integers of length equal to dimension
#        :param collisionSpaceTransformation:
#                    EquationCollection object, defining a transformation of distribution space (pdfs)
#                    into collision space. Collision space can be for example a moment or cumulant representation.
#                    The equation collection must have as many free symbols as there are stencil entries.
#                    The free symbols are termed collisionSpaceSymbols and are passed to the relaxation scheme.
#        :param relaxationScheme:
#                    a relaxation scheme object, providing information about the relaxation process and the
#                    equilibrium in collision space
#        :param inverseCollisionSpaceTransformation:
#                    if passed None, the inverse transformation is determined by
#                    inverting the collisionSpaceTransformation using sympy. There are
#                    cases where sympy is slow or unable to do that, in this case the
#                    inverse can be specified here.
#        """
#
#    @property
#    def stencil(self):
#        pass # TODO
#
#    #@property
#    #def compressible(self):
#    #    pass # TODO
#
#    @property
#    def zeroCentered(self):
#        pass
#
#    @property
#    def dim(self):
#        return len(self.stencil[0])
#
#    def preCollisionSymbols(self):
#        return sp.symbols("f_:%d" % (len(self.stencil),))
#
#    def collisionSpaceSymbols(self):
#        return [sp.Symbol("c_%d" % (i,)) for i in range(len(self.stencil))]
#
#    def postCollisionSymbols(self):
#        return [sp.Symbol("d_%d" % (i,)) for i in range(len(self.stencil))]
#
#    def equilibrium(self):
#        """Returns equation collection, defining the equilibrium in collision space"""
#
#    def conservedQuantities(self):
#        """Returns equation collection defining conserved quantities"""
#        pass
#
#    def transformToCollisionSpace(self, equationCollection):
#        pass
#
#    def transformToPdfSpace(self, equationCollection):
#        pass
