import ipywidgets.widgets as widgets
from IPython.display import display
from ipywidgets.widgets import FloatText, Label, VBox, HBox, Select, BoundedFloatText, Button
from lbmpy.relaxationrates import relaxationRateFromLatticeViscosity, latticeViscosityFromRelaxationRate


class ScalingWidget:
    def __init__(self):
        self.scalingType = Select(options=[r'diffusive (fixed relaxation rate)',
                                           r'acoustic (fixed dt)',
                                           r'fixed lattice velocity'])
        self.physicalLength = FloatText(value=0.1)
        self.cellsPerLength = FloatText(value=256.0)
        self.maxPhysicalVelocity = FloatText(value=0.01)
        self.dx = FloatText(value=self._getDx())
        self.kinematicViscosity = BoundedFloatText(value=1.0, min=0, max=1e12)
        self.omega = BoundedFloatText(min=0, max=2, value=1.9)
        self.dt = FloatText(value=0)
        self.maxLatticeVelocity = FloatText(disabled=True)
        self.re = FloatText(disabled=True)

        self.processingUpdate = False

        def makeLabel(text):
            labelLayout = {'width': '200px'}
            return Label(value=text, layout=labelLayout)

        def makeButtons(inputWidget, inverse=False):
            buttonLayout = {'width': '20px'}
            factor = 0.5 if inverse else 2.0
            doubleBtn = Button(description="", button_style='warning', layout=buttonLayout)
            halfBtn = Button(description="", button_style='success', layout=buttonLayout)

            def doubleValue(v):
                inputWidget.value *= factor

            def halfValue(v):
                inputWidget.value /= factor

            widgets.jslink((doubleBtn, 'disabled'), (inputWidget, 'disabled'))
            widgets.jslink((halfBtn, 'disabled'), (inputWidget, 'disabled'))
            doubleBtn.on_click(doubleValue)
            halfBtn.on_click(halfValue)
            return [halfBtn, doubleBtn]

        self.form = VBox([
            HBox([makeLabel(r'Scaling'), self.scalingType]),
            HBox([makeLabel(r"Physical Length $[m]$"), self.physicalLength]),
            HBox([makeLabel(r"Max. velocity $[m/s]$"), self.maxPhysicalVelocity]),
            HBox([makeLabel(r"Cells per length"), self.cellsPerLength] + makeButtons(self.cellsPerLength, True)),
            HBox([makeLabel(r"dx"), self.dx] + makeButtons(self.dx)),
            HBox([makeLabel(r"Kinematic viscosity $10^{-6}[m^2/s]$"), self.kinematicViscosity]),
            HBox([makeLabel(r"Relaxation rate $\omega$"), self.omega]),
            HBox([makeLabel(r"dt"), self.dt] + makeButtons(self.dt)),
            HBox([makeLabel(r"Max. lattice velocity"), self.maxLatticeVelocity]),
            HBox([makeLabel(r"Re"), self.re]),
        ])

        # Change listeners
        self.physicalLength.observe(self._on_physicalLength_change, names='value')
        self.cellsPerLength.observe(self._on_cellsPerLength_change, names='value')
        self.dx.observe(self._on_dx_change, names='value')
        self.physicalLength.observe(self._updateRe)
        self.kinematicViscosity.observe(self._updateRe)
        self.maxPhysicalVelocity.observe(self._updateRe)

        for obj in [self.scalingType, self.kinematicViscosity, self.omega, self.dt, self.maxLatticeVelocity]:
            obj.observe(self._updateFreeParameter, names='value')

        self._updateFreeParameter()
        self._updateRe()

    def _getDx(self):
        return self.physicalLength.value / self.cellsPerLength.value

    def _updateDtFromRelaxationRateViscosityAndDx(self):
        if self.omega.value == 0:
            return 0
        latticeViscosity = latticeViscosityFromRelaxationRate(self.omega.value)
        self.dt.value = latticeViscosity / (self.kinematicViscosity.value * 1e-6) * self.dx.value ** 2

    def _updateDtFromDxAndLatticeVelocity(self):
        if self.maxPhysicalVelocity.value == 0:
            return
        self.dt.value = self.maxLatticeVelocity.value / self.maxPhysicalVelocity.value * self.dx.value

    def _updateOmegaFromViscosityAndDtDx(self):
        if self.dx.value == 0:
            return
        latticeViscosity = self.kinematicViscosity.value * 1e-6 * self.dt.value / (self.dx.value ** 2)
        self.omega.value = relaxationRateFromLatticeViscosity(latticeViscosity)

    def _updateFreeParameter(self, change=None):
        self.dt.disabled = True
        self.omega.disabled = True
        self.maxLatticeVelocity.disabled = True

        if self.scalingType.value == r'diffusive (fixed relaxation rate)':
            self.omega.disabled = False
            self._updateDtFromRelaxationRateViscosityAndDx()
            self._updateLatticeVelocityFromDxDtAndPhysicalVelocity()
        elif self.scalingType.value == r'acoustic (fixed dt)':
            self._updateOmegaFromViscosityAndDtDx()
            self._updateDtFromDxAndLatticeVelocity()
            self.maxLatticeVelocity.disabled = False
        elif self.scalingType.value == r'fixed lattice velocity':
            self._updateOmegaFromViscosityAndDtDx()
            self.dt.disabled = False
            self._updateLatticeVelocityFromDxDtAndPhysicalVelocity()
        else:
            raise ValueError("Unknown Scaling Type")

    def _updateLatticeVelocityFromDxDtAndPhysicalVelocity(self):
        if self.dx.value == 0:
            return
        self.maxLatticeVelocity.value = self.dt.value / self.dx.value * self.maxPhysicalVelocity.value

    def _updateRe(self, change=None):
        if self.kinematicViscosity.value == 0:
            return
        L = self.physicalLength.value
        u = self.maxPhysicalVelocity.value
        nu = self.kinematicViscosity.value * 1e-6
        self.re.value = round(L * u / nu, 7)

    def _on_dx_change(self, change):
        if self.processingUpdate:
            return
        if self.dx.value == 0:
            return
        self.processingUpdate = True
        self.cellsPerLength.value = self.physicalLength.value / self.dx.value
        self._updateFreeParameter()
        self.processingUpdate = False

    def _on_cellsPerLength_change(self, change):
        if self.processingUpdate:
            return
        if self.cellsPerLength.value == 0:
            return
        self.processingUpdate = True
        self.dx.value = self.physicalLength.value / self.cellsPerLength.value
        self._updateFreeParameter()
        self.processingUpdate = False

    def _on_physicalLength_change(self, change):
        if self.cellsPerLength.value == 0:
            return
        self.dx.value = self.physicalLength.value / self.cellsPerLength.value
        self._updateFreeParameter()

    def show(self):
        from IPython.display import HTML
        display(HTML("""
        <style>
            button[disabled], html input[disabled] {
                background-color: #eaeaea !important;
            }
        </style>
        """))
        return self.form
