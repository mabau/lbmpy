import ipywidgets.widgets as widgets
from IPython.display import display
from ipywidgets.widgets import FloatText, Label, VBox, HBox, Select, BoundedFloatText, Button
from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity, lattice_viscosity_from_relaxation_rate


class ScalingWidget:
    def __init__(self):
        self.scaling_type = Select(options=[r'diffusive (fixed relaxation rate)',
                                            r'acoustic (fixed dt)',
                                            r'fixed lattice velocity'])
        self.physical_length = FloatText(value=0.1)
        self.cells_per_length = FloatText(value=256.0)
        self.max_physical_velocity = FloatText(value=0.01)
        self.dx = FloatText(value=self._get_dx())
        self.kinematic_viscosity = BoundedFloatText(value=1.0, min=0, max=1e12)
        self.omega = BoundedFloatText(min=0, max=2, value=1.9)
        self.dt = FloatText(value=0)
        self.max_lattice_velocity = FloatText(disabled=True)
        self.re = FloatText(disabled=True)

        self.processing_update = False

        def make_label(text):
            label_layout = {'width': '200px'}
            return Label(value=text, layout=label_layout)

        def make_buttons(input_widget, inverse=False):
            button_layout = {'width': '20px'}
            factor = 0.5 if inverse else 2.0
            double_btn = Button(description="", button_style='warning', layout=button_layout)
            half_btn = Button(description="", button_style='success', layout=button_layout)

            def double_value(_):
                input_widget.value *= factor

            def half_value(_):
                input_widget.value /= factor

            widgets.jslink((double_btn, 'disabled'), (input_widget, 'disabled'))
            widgets.jslink((half_btn, 'disabled'), (input_widget, 'disabled'))
            double_btn.on_click(double_value)
            half_btn.on_click(half_value)
            return [half_btn, double_btn]

        self.form = VBox([
            HBox([make_label(r'Scaling'), self.scaling_type]),
            HBox([make_label(r"Physical Length $[m]$"), self.physical_length]),
            HBox([make_label(r"Max. velocity $[m/s]$"), self.max_physical_velocity]),
            HBox([make_label(r"Cells per length"), self.cells_per_length] + make_buttons(self.cells_per_length, True)),
            HBox([make_label(r"dx"), self.dx] + make_buttons(self.dx)),
            HBox([make_label(r"Kinematic viscosity $10^{-6}[m^2/s]$"), self.kinematic_viscosity]),
            HBox([make_label(r"Relaxation rate $\omega$"), self.omega]),
            HBox([make_label(r"dt"), self.dt] + make_buttons(self.dt)),
            HBox([make_label(r"Max. lattice velocity"), self.max_lattice_velocity]),
            HBox([make_label(r"Re"), self.re]),
        ])

        # Change listeners
        self.physical_length.observe(self._on_physical_length_change, names='value')
        self.cells_per_length.observe(self._on_cells_per_length_change, names='value')
        self.dx.observe(self._on_dx_change, names='value')
        self.physical_length.observe(self._update_re)
        self.kinematic_viscosity.observe(self._update_re)
        self.max_physical_velocity.observe(self._update_re)

        for obj in [self.scaling_type, self.kinematic_viscosity, self.omega, self.dt, self.max_lattice_velocity]:
            obj.observe(self._update_free_parameter, names='value')

        self._update_free_parameter()
        self._update_re()

    def _get_dx(self):
        return self.physical_length.value / self.cells_per_length.value

    def _update_dt_from_relaxation_rate_viscosity_and_dx(self):
        if self.omega.value == 0:
            return 0
        lattice_viscosity = lattice_viscosity_from_relaxation_rate(self.omega.value)
        self.dt.value = lattice_viscosity / (self.kinematic_viscosity.value * 1e-6) * self.dx.value ** 2

    def _update_dt_from_dx_and_lattice_velocity(self):
        if self.max_physical_velocity.value == 0:
            return
        self.dt.value = self.max_lattice_velocity.value / self.max_physical_velocity.value * self.dx.value

    def _update_omega_from_viscosity_and_dt_dx(self):
        if self.dx.value == 0:
            return
        lattice_viscosity = self.kinematic_viscosity.value * 1e-6 * self.dt.value / (self.dx.value ** 2)
        self.omega.value = relaxation_rate_from_lattice_viscosity(lattice_viscosity)

    def _update_free_parameter(self, _=None):
        self.dt.disabled = True
        self.omega.disabled = True
        self.max_lattice_velocity.disabled = True

        if self.scaling_type.value == r'diffusive (fixed relaxation rate)':
            self.omega.disabled = False
            self._update_dt_from_relaxation_rate_viscosity_and_dx()
            self._update_lattice_velocity_from_dx_dt_and_physical_velocity()
        elif self.scaling_type.value == r'acoustic (fixed dt)':
            self._update_omega_from_viscosity_and_dt_dx()
            self._update_dt_from_dx_and_lattice_velocity()
            self.max_lattice_velocity.disabled = False
        elif self.scaling_type.value == r'fixed lattice velocity':
            self._update_omega_from_viscosity_and_dt_dx()
            self.dt.disabled = False
            self._update_lattice_velocity_from_dx_dt_and_physical_velocity()
        else:
            raise ValueError("Unknown Scaling Type")

    def _update_lattice_velocity_from_dx_dt_and_physical_velocity(self):
        if self.dx.value == 0:
            return
        self.max_lattice_velocity.value = self.dt.value / self.dx.value * self.max_physical_velocity.value

    def _update_re(self, _=None):
        if self.kinematic_viscosity.value == 0:
            return
        L = self.physical_length.value
        u = self.max_physical_velocity.value
        nu = self.kinematic_viscosity.value * 1e-6
        self.re.value = round(L * u / nu, 7)

    def _on_dx_change(self, _):
        if self.processing_update:
            return
        if self.dx.value == 0:
            return
        self.processing_update = True
        self.cells_per_length.value = self.physical_length.value / self.dx.value
        self._update_free_parameter()
        self.processing_update = False

    def _on_cells_per_length_change(self, _):
        if self.processing_update:
            return
        if self.cells_per_length.value == 0:
            return
        self.processing_update = True
        self.dx.value = self.physical_length.value / self.cells_per_length.value
        self._update_free_parameter()
        self.processing_update = False

    def _on_physical_length_change(self, _):
        if self.cells_per_length.value == 0:
            return
        self.dx.value = self.physical_length.value / self.cells_per_length.value
        self._update_free_parameter()

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
