import sympy as sp
from collections import Counter
from pystencils.transformations import fastSubs
from lbmpy.equilibria import maxwellBoltzmannEquilibrium
from lbmpy.latticemodel import LatticeModel, LbmCollisionRule
from lbmpy.cumulants import cumulantsFromPdfs, getDefaultIndexedSymbols, rawMomentsFromCumulants, continuousCumulant
from lbmpy.moments import momentsUpToComponentOrder, momentMatrix
from lbmpy.equilibria import getWeights
from lbmpy.densityVelocityExpressions import getDensityVelocityExpressions


class SimpleBoltzmannRelaxation:
    def __init__(self, omega):
        self._omega = omega
        self.addPostCollisionsAsSubexpressions = False

    def __call__(self, preCollisionSymbols, indices):
        pre = {a: b for a, b in zip(indices, preCollisionSymbols)}
        post = {}
        dim = len(indices[0])

        # conserved quantities
        for idx, value in pre.items():
            if sum(idx) == 0 or sum(idx) == 1:
                post[idx] = pre[idx]

        # hydrodynamic relaxation
        for idx in indices:
            idxCounter = Counter(idx)
            if len(idxCounter.keys() - set([0, 1])) == 0 and idxCounter[1] == 2:
                post[idx] = (1 - self._omega) * pre[idx]

        # set remaining values to their equilibrium value (i.e. relaxationRate=1)
        maxwellBoltzmann = maxwellBoltzmannEquilibrium(dim, c_s_sq=sp.Rational(1, 3))
        for idx, value in pre.items():
            if idx not in post:
                post[idx] = continuousCumulant(maxwellBoltzmann, idx)

        return [post[idx] for idx in indices]


class CorrectedD3Q27Collision:
    
    def __init__(self, omegaArr):
        self.omega = omegaArr
        self.addPostCollisionsAsSubexpressions = True

    def __call__(self, preCollisionSymbols, indices):
        assert len(indices) == 27

        pre = {a: b for a, b in zip(indices, preCollisionSymbols)}
        post = {}
    
        ux, uy, uz = pre[(1, 0, 0)], pre[(0, 1, 0)], pre[(0, 0, 1)]
        rho = sp.exp(pre[(0, 0, 0)])
    
        post[(0, 0, 0)] = pre[(0, 0, 0)]
        post[(1, 0, 0)] = pre[(1, 0, 0)]
        post[(0, 1, 0)] = pre[(0, 1, 0)]
        post[(0, 0, 1)] = pre[(0, 0, 1)]
        post[(1, 1, 0)] = 1 - self.omega[0] * pre[(1, 1, 0)]
        post[(1, 0, 1)] = 1 - self.omega[0] * pre[(1, 0, 1)]
        post[(0, 1, 1)] = 1 - self.omega[0] * pre[(0, 1, 1)]
    
        Dxux = - self.omega[0] / 2 / rho * (2 * pre[(2,0,0)] - pre[(0,2,0)] - pre[(0,0,2)]) - self.omega[1] / 2 / rho * (pre[(2,0,0)] + pre[(0,2,0)] + pre[(0,0,2)] - pre[(0,0,0)])
        Dyuy = Dxux + 3 * self.omega[0] / rho / 2 * (pre[(2, 0, 0)] - pre[(0, 2, 0)])
        Dzuz = Dxux + 3 * self.omega[0] / rho / 2 * (pre[(2, 0, 0)] - pre[(0, 0, 2)])
    
        CS_200__m__CS020 = (1-self.omega[0]) * (pre[(2,0,0)] - pre[(0,2,0)]) - 3 * rho * (1 - self.omega[0] / 2) * (Dxux * ux**2 - Dyuy * uy**2)
        CS_200__m__CS002 = (1-self.omega[0]) * (pre[(2,0,0)] - pre[(0,0,2)]) - 3 * rho * (1 - self.omega[0] / 2) * (Dxux * ux**2 - Dzuz * uz**2)
        CS_200__p__CS020__p__CS_002 = self.omega[1] * pre[(0,0,0)] + (1-self.omega[1]) * (pre[(2,0,0)] + pre[(0,2,0)] + pre[(0,0,2)]) - 3 * rho *(1 - self.omega[1]/2) * (Dxux * ux**2 + Dyuy * uy**2 + Dzuz * uz**2)
        post[(2, 0, 0)] = (CS_200__m__CS020 + CS_200__m__CS002 + CS_200__p__CS020__p__CS_002) / 3
        post[(0, 2, 0)] = post[(2, 0, 0)] - CS_200__m__CS020
        post[(0, 0, 2)] = post[(2, 0, 0)] - CS_200__m__CS002
    
        CS_120__p__CS_102 = (1 - self.omega[2]) * (pre[(1, 2, 0)] + pre[(1, 0, 2)])
        CS_210__p__CS_012 = (1 - self.omega[2]) * (pre[(2, 1, 0)] + pre[(0, 1, 2)])
        CS_201__p__CS_021 = (1 - self.omega[2]) * (pre[(2, 0, 1)] + pre[(0, 2, 1)])
        CS_120__m__CS_102 = (1 - self.omega[3]) * (pre[(1, 2, 0)] - pre[(1, 0, 2)])
        CS_210__m__CS_012 = (1 - self.omega[3]) * (pre[(2, 1, 0)] - pre[(0, 1, 2)])
        CS_201__m__CS_021 = (1 - self.omega[3]) * (pre[(2, 0, 1)] - pre[(0, 2, 1)])
    
        post[(1, 2, 0)] = (CS_120__p__CS_102 + CS_120__m__CS_102) / 2
        post[(1, 0, 2)] = (CS_120__p__CS_102 - CS_120__m__CS_102) / 2
        post[(0, 1, 2)] = (CS_210__p__CS_012 - CS_210__m__CS_012) / 2
        post[(2, 1, 0)] = (CS_210__p__CS_012 + CS_210__m__CS_012) / 2
        post[(2, 0, 1)] = (CS_201__p__CS_021 + CS_201__m__CS_021) / 2
        post[(0, 2, 1)] = (CS_201__p__CS_021 - CS_201__m__CS_021) / 2
        post[(1, 1, 1)] = (1 - self.omega[4]) * pre[(1, 1, 1)]
    
        CS_220__m__2CS_202__p__CS_022 = (1 - self.omega[5]) * (pre[(2, 2, 0)] - 2 * pre[(2, 0, 2)] + pre[(0, 2, 2)])
        CS_220__p__CS_202__m__2CS_022 = (1 - self.omega[5]) * (pre[(2, 2, 0)] + pre[(2, 0, 2)] - 2 * pre[(0, 2, 2)])
        CS_220__p__CS_202__p__CS_022 = (1 - self.omega[6]) * (pre[(2, 2, 0)] + pre[(2, 0, 2)] + pre[(0, 2, 2)])
    
        post[(2, 2, 0)] = (CS_220__m__2CS_202__p__CS_022 + CS_220__p__CS_202__m__2CS_022 + CS_220__p__CS_202__p__CS_022) / 3
        post[(2, 0, 2)] = (CS_220__p__CS_202__p__CS_022 - CS_220__m__2CS_202__p__CS_022) / 3
        post[(0, 2, 2)] = (CS_220__p__CS_202__p__CS_022 - CS_220__p__CS_202__m__2CS_022) / 3
    
        post[(2, 1, 1)] = (1 - self.omega[7]) * pre[(2, 1, 1)]
        post[(1, 2, 1)] = (1 - self.omega[7]) * pre[(1, 2, 1)]
        post[(1, 1, 2)] = (1 - self.omega[7]) * pre[(1, 1, 2)]
        post[(2, 2, 1)] = (1 - self.omega[8]) * pre[(2, 2, 1)]
        post[(2, 1, 2)] = (1 - self.omega[8]) * pre[(2, 1, 2)]
        post[(1, 2, 2)] = (1 - self.omega[8]) * pre[(1, 2, 2)]
        post[(2, 2, 2)] = (1 - self.omega[9]) * pre[(2, 2, 2)]
        return [post[idx] for idx in indices]


class CumulantRelaxationLatticeModel(LatticeModel):

    def __init__(self, stencil, cumulantCollision, forceModel=None):
        super(CumulantRelaxationLatticeModel, self).__init__(stencil, True, forceModel)
        self.cumulantCollision = cumulantCollision

    @property
    def weights(self):
        return getWeights(self._stencil)

    def getCollisionRule(self):
        pdfSymbols = tuple(self.pdfSymbols)
        moments = tuple(momentsUpToComponentOrder(2, dim=self.dim))

        densityIdx = (0, 0) if self.dim == 2 else (0, 0, 0)
        velIndices = [(1, 0), (0, 1)] if self.dim == 2 else [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        momentSymbols = tuple(getDefaultIndexedSymbols(None, "m", moments))
        cumulantSymbols = getDefaultIndexedSymbols(None, "c", moments)
        cumulantSymbols[moments.index(densityIdx)] = sp.ln(self.symbolicDensity)
        for u_i, idx in zip(self.symbolicVelocity, velIndices):
            cumulantSymbols[moments.index(idx)] = u_i
        cumulantSymbols = tuple(cumulantSymbols)
        cumFromPdf = cumulantsFromPdfs(self.stencil, pdfSymbols, cumulantSymbols)

        rhoSubexprs, rhoEq, uSubexprs, uEqs = getDensityVelocityExpressions(self.stencil, self.pdfSymbols,
                                                                            self.compressible)

        # for some force models the velocity has to be shifted
        if self.forceModel and hasattr(self.forceModel, "equilibriumVelocity"):
            uSymbols = self.symbolicVelocity
            uRhs = [e.rhs for e in uEqs]
            correctedVel = self.forceModel.equilibriumVelocity(self, uRhs, self.symbolicDensity)
            uEqs = [sp.Eq(u_i, correctedVel_i) for u_i, correctedVel_i in zip(uSymbols, correctedVel)]

        subExpressions = rhoSubexprs + [rhoEq] + uSubexprs + uEqs
        subExpressions += [eq for eq, moment in zip(cumFromPdf, moments)
                           if moment not in velIndices and moment != densityIdx]

        collidedValues = self.cumulantCollision(cumulantSymbols, moments)
        if self.cumulantCollision.addPostCollisionsAsSubexpressions:
            postcollisionSymbols = getDefaultIndexedSymbols(None, "cp", moments)
            subExpressions += [sp.Eq(s, cp) for s, cp in zip(postcollisionSymbols, collidedValues)]
            collidedValues = postcollisionSymbols

        momFromCum = rawMomentsFromCumulants(self.stencil, cumulantSymbols, momentSymbols)
        substitutions = {cumulantSymbol: collidedValue
                         for cumulantSymbol, collidedValue in zip(cumulantSymbols, collidedValues)}
        collidedMoments = [fastSubs(rawMomentEq.rhs, substitutions) for rawMomentEq in momFromCum]
        M = momentMatrix(moments, self.stencil)
        collisionResult = M.inv() * sp.Matrix(collidedMoments)

        if self.forceModel:
            collisionResult += sp.Matrix(self.forceModel(latticeModel=self))
        collisionEqs = [sp.Eq(dst_i, t) for dst_i, t in zip(self.pdfDestinationSymbols, collisionResult)]

        return LbmCollisionRule(collisionEqs, subExpressions, self)
