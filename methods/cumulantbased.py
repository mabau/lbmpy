import sympy as sp
from collections import OrderedDict
from lbmpy.methods.abstractlbmethod import AbstractLbMethod, LbmCollisionRule, RelaxationInfo
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation
from lbmpy.moments import MOMENT_SYMBOLS
from lbmpy.cumulants import cumulantsFromPdfs


class CumulantBasedLbMethod(AbstractLbMethod):

    def __init__(self, stencil, cumulantToRelaxationInfoDict, conservedQuantityComputation, forceModel=None):
        assert isinstance(conservedQuantityComputation, AbstractConservedQuantityComputation)
        super(CumulantBasedLbMethod, self).__init__(stencil)

        self._forceModel = forceModel
        self._cumulantToRelaxationInfoDict = OrderedDict(cumulantToRelaxationInfoDict.items())
        self._conservedQuantityComputation = conservedQuantityComputation

    @property
    def cumulantToRelaxationInfoDict(self):
        return self._cumulantToRelaxationInfoDict

    def setFirstMomentRelaxationRate(self, relaxationRate):
        for e in MOMENT_SYMBOLS[:self.dim]:
            assert e in self._cumulantToRelaxationInfoDict, "First cumulants are not relaxed separately by this method"
        for e in MOMENT_SYMBOLS[:self.dim]:
            prevEntry = self._cumulantToRelaxationInfoDict[e]
            newEntry = RelaxationInfo(prevEntry[0], relaxationRate)
            self._cumulantToRelaxationInfoDict[e] = newEntry

    @property
    def moments(self):
        return tuple(self._cumulantToRelaxationInfoDict.keys())

    @property
    def momentEquilibriumValues(self):
        return tuple([e.equilibriumValue for e in self._cumulantToRelaxationInfoDict.values()])

    @property
    def relaxationRates(self):
        return tuple([e.relaxationRate for e in self._cumulantToRelaxationInfoDict.values()])

    def _repr_html_(self):
        table = """
        <table style="border:none; width: 100%">
            <tr {nb}>
                <th {nb} >Cumulant</th>
                <th {nb} >Eq. Value </th>
                <th {nb} >Relaxation Time</th>
            </tr>
            {content}
        </table>
        """
        content = ""
        for cumulant, (eqValue, rr) in self._cumulantToRelaxationInfoDict.items():
            vals = {
                'rr': sp.latex(rr),
                'cumulant': sp.latex(cumulant),
                'eqValue': sp.latex(eqValue),
                'nb': 'style="border:none"',
            }
            content += """<tr {nb}>
                            <td {nb}>${cumulant}$</td>
                            <td {nb}>${eqValue}$</td>
                            <td {nb}>${rr}$</td>
                         </tr>\n""".format(**vals)
        return table.format(content=content, nb='style="border:none"')

    def getEquilibrium(self, conservedQuantityEquations=None):
        D = sp.eye(len(self.relaxationRates))
        return self._getCollisionRuleWithRelaxationMatrix(D, conservedQuantityEquations=conservedQuantityEquations)

    def getCollisionRule(self, conservedQuantityEquations=None):
        # Step 1: transform input into cumulant space

        cumulantsFromPdfs(self.stencil, )
        # Step 2: create linear transformation from basic cumulants to requested cumulants

        # Step 3: relaxation

        # Step 4: transform back into standard cumulants

        # Step 5: transform cumulants to moments

        # Step 6: transform moments to pdfs

        D = sp.diag(*self.relaxationRates)
        relaxationRateSubExpressions, D = self._generateRelaxationMatrix(D)
        eqColl = self._getCollisionRuleWithRelaxationMatrix(D, relaxationRateSubExpressions, conservedQuantityEquations)
        if self._forceModel is not None:
            forceModelTerms = self._forceModel(self)
            newEqs = [sp.Eq(eq.lhs, eq.rhs + fmt) for eq, fmt in zip(eqColl.mainEquations, forceModelTerms)]
            eqColl = eqColl.copy(newEqs)
        return eqColl

