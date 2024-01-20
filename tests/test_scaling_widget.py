from lbmpy.parameterization import Scaling, ScalingWidget


def test_scaling_widget():
    w = ScalingWidget()
    s = Scaling(physical_length=16, physical_velocity=0.001, kinematic_viscosity=1e-6, cells_per_length=8192)

    w.scaling_type.value = 'diffusive (fixed relaxation rate)'
    w.physical_length.value = 16
    w.max_physical_velocity.value = 0.001
    w.cells_per_length.value = 8192
    w.kinematic_viscosity.value = 1
    w.omega.value = 1.9

    scaling_result = s.diffusive_scaling(1.9)

    assert w.dx.value == 16 / 8192 and s.dx == w.dx.value
    assert round(w.dt.value, 5) == 0.03346 and scaling_result.dt == w.dt.value
    assert round(w.max_lattice_velocity.value, 5) == 0.01713
    assert scaling_result.lattice_velocity == w.max_lattice_velocity.value
    assert w.re.value == 16000 and s.reynolds_number == w.re.value

    s = Scaling(physical_length=1, physical_velocity=0.001, kinematic_viscosity=2e-6, cells_per_length=4096)
    w.scaling_type.value = 'fixed lattice velocity'
    w.physical_length.value = 1
    w.max_physical_velocity.value = 0.001
    w.cells_per_length.value = 4096
    w.kinematic_viscosity.value = 2
    w.max_lattice_velocity.value = 0.1
    scaling_result = s.fixed_lattice_velocity_scaling(lattice_velocity=0.1)

    assert round(w.omega.value, 2) == 0.34 and scaling_result.relaxation_rate == w.omega.value
    assert round(w.dt.value, 4) == 0.0244 and scaling_result.dt == w.dt.value
    assert round(w.re.value) == 500 and s.reynolds_number == w.re.value

    s = Scaling(physical_length=2, physical_velocity=0.002, kinematic_viscosity=4e-6, cells_per_length=2048)
    w.scaling_type.value = 'acoustic (fixed dt)'
    w.physical_length.value = 2
    w.max_physical_velocity.value = 0.002
    w.cells_per_length.value = 2048
    w.kinematic_viscosity.value = 4
    w.dt.value = 0.01
    scaling_result = s.acoustic_scaling(dt=0.01)

    assert round(w.omega.value, 2) == 1.60 and scaling_result.relaxation_rate == w.omega.value
    assert w.max_lattice_velocity.value == 0.02048 and scaling_result.lattice_velocity == w.max_lattice_velocity.value
    assert round(w.re.value) == 1000 and s.reynolds_number == w.re.value
