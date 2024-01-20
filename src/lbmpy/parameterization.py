from collections import namedtuple

import ipywidgets.widgets as widgets
from IPython.display import display
from ipywidgets.widgets import BoundedFloatText, Button, FloatText, HBox, Label, Select, VBox

from lbmpy.relaxationrates import (
    lattice_viscosity_from_relaxation_rate, relaxation_rate_from_lattice_viscosity)


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
            self.dt.disabled = False
            self._update_lattice_velocity_from_dx_dt_and_physical_velocity()
        elif self.scaling_type.value == r'fixed lattice velocity':
            self._update_omega_from_viscosity_and_dt_dx()
            self._update_dt_from_dx_and_lattice_velocity()
            self.max_lattice_velocity.disabled = False
        else:
            raise ValueError("Unknown Scaling Type")

    def _update_lattice_velocity_from_dx_dt_and_physical_velocity(self):
        if self.dx.value == 0:
            return
        self.max_lattice_velocity.value = self.dt.value / self.dx.value * self.max_physical_velocity.value

    def _update_re(self, _=None):
        if self.kinematic_viscosity.value == 0:
            return
        viscosity = self.kinematic_viscosity.value * 1e-6
        re = reynolds_number(self.physical_length.value, self.max_physical_velocity.value, viscosity)
        self.re.value = round(re, 7)

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


def reynolds_number(length, velocity, kinematic_viscosity):
    """Computes the Reynolds number.

    All arguments have to be in the same set of units, for example all in SI units.
    """
    return length / kinematic_viscosity * velocity


class Scaling:
    """Class to convert physical parameters into lattice units.

    The scaling is created with the central physical parameters length, velocity and viscosity
    These parameters fix the Reynolds number.

    Grid information is specified by fixing the number of cells to resolve the physical length.
    The three lattice parameters relaxation rate, lattice velocity and time step length can be computed, if one of
    them is given (see the '*_scaling' methods, that receive one parameter and compute the other two.

    Args:
        physical_length: typical physical length [m]
        physical_velocity: (maximum) physical velocity [m/s]
                            usually the maximum velocity occurring in the simulation domain should be passed here,
                            such that the fixed lattice velocity scaling can ensure a maximum lattice velocity
        kinematic_viscosity: kinematic viscosity in physical units [m*m/s]
        cells_per_length: number of cells to resolve the physical length with
    """
    def __init__(self, physical_length, physical_velocity, kinematic_viscosity, cells_per_length):
        self._physical_length = physical_length
        self._physical_velocity = physical_velocity
        self._kinematic_viscosity = kinematic_viscosity
        self.cells_per_length = cells_per_length
        self._reynolds_number = None
        self._parameter_update()

    def _parameter_update(self):
        self._reynolds_number = reynolds_number(self._physical_length, self._physical_velocity,
                                                self._kinematic_viscosity)

    @property
    def dx(self):
        return self.physical_length / self.cells_per_length

    @property
    def reynolds_number(self):
        return self._reynolds_number

    @property
    def physical_velocity(self):
        return self._physical_velocity

    @physical_velocity.setter
    def physical_velocity(self, val):
        self._physical_velocity = val
        self._parameter_update()

    @property
    def kinematic_viscosity(self):
        return self._kinematic_viscosity

    @kinematic_viscosity.setter
    def kinematic_viscosity(self, val):
        self._kinematic_viscosity = val
        self._parameter_update()

    @property
    def physical_length(self):
        return self._physical_length

    @physical_length.setter
    def physical_length(self, val):
        self._physical_length = val
        self._parameter_update()

    def fixed_lattice_velocity_scaling(self, lattice_velocity=0.1):
        """Computes relaxation rate and time step length from lattice velocity.

        Check the returned relaxation rate! If it is very close to 2, the simulation might get unstable.
        In that case increase the resolution (more cells_per_length)
        All physical quantities have to be passed in the same set of units.

        Args:
            lattice_velocity: Lattice velocity corresponding to physical_velocity.
                                  Maximum velocity should not be larger than 0.1 i.e. the fluid should not move
                                  faster than 0.1 cells per time step

        Returns:
            relaxation_rate, dt
        """
        dt = lattice_velocity / self.physical_velocity * self.dx
        lattice_viscosity = self.kinematic_viscosity * dt / (self.dx ** 2)
        relaxation_rate = relaxation_rate_from_lattice_viscosity(lattice_viscosity)
        ResultType = namedtuple("FixedLatticeVelocityScalingResult", ['relaxation_rate', 'dt'])
        return ResultType(relaxation_rate, dt)

    def diffusive_scaling(self, relaxation_rate):
        """Computes time step length and lattice velocity from relaxation rate

        For stable simulations the lattice velocity should be smaller than 0.1

        Args:
            relaxation_rate: relaxation rate (between 0 and 2)

        Returns:
            dt, lattice velocity
        """
        lattice_viscosity = lattice_viscosity_from_relaxation_rate(relaxation_rate)
        dx = self.physical_length / self.cells_per_length
        dt = lattice_viscosity / self.kinematic_viscosity * dx ** 2
        lattice_velocity = dt / dx * self.physical_velocity
        ResultType = namedtuple("DiffusiveScalingResult", ['dt', 'lattice_velocity'])
        return ResultType(dt, lattice_velocity)

    def acoustic_scaling(self, dt):
        """Computes relaxation rate and lattice velocity from time step length.

        Args:
            dt: time step length

        Returns:
            relaxation_rate, lattice_velocity
        """
        lattice_velocity = dt / self.dx * self.physical_velocity
        lattice_viscosity = self.kinematic_viscosity * dt / (self.dx ** 2)
        relaxation_rate = relaxation_rate_from_lattice_viscosity(lattice_viscosity)
        ResultType = namedtuple("AcousticScalingResult", ['relaxation_rate', 'lattice_velocity'])
        return ResultType(relaxation_rate, lattice_velocity)
