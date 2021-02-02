from lbmpy.forcemodels import default_velocity_shift


#   =========================== Centered Cumulant Force Model ==========================================================


class CenteredCumulantForceModel:
    """
        A force model to be used with the centered cumulant-based LB Method.
        It only shifts the macroscopic and equilibrium velocities and does not 
        introduce a forcing term to the collision process. Forcing is then applied 
        through relaxation of the first central moments in the shifted frame of 
        reference (cf. https://doi.org/10.1016/j.camwa.2015.05.001).
    """

    def __init__(self, force):
        self._force = force
        self.override_momentum_relaxation_rate = 2

    def __call__(self, lb_method, **kwargs):
        raise Exception('This force model does not provide a forcing term.')

    def macroscopic_velocity_shift(self, density):
        return default_velocity_shift(density, self._force)

    def equilibrium_velocity_shift(self, density):
        return default_velocity_shift(density, self._force)
