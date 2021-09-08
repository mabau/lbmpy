from lbmpy.forcemodels import AbstractForceModel, default_velocity_shift


#   =========================== Centered Cumulant Force Model ==========================================================


class CenteredCumulantForceModel(AbstractForceModel):
    """
    A force model to be used with the centered cumulant-based LB Method.
    It only shifts the macroscopic and equilibrium velocities and does not introduce a forcing term to the
    collision process. Forcing is then applied through relaxation of the first central moments in the shifted frame of
    reference (cf. https://doi.org/10.1016/j.camwa.2015.05.001).

    Args:
        force: force vector which should be applied to the fluid
    """

    def __init__(self, force):
        self.override_momentum_relaxation_rate = 2

        super(CenteredCumulantForceModel, self).__init__(force)

    def __call__(self, lb_method):
        raise Exception('This force model does not provide a forcing term.')

    def equilibrium_velocity_shift(self, density):
        return default_velocity_shift(density, self.symbolic_force_vector)
