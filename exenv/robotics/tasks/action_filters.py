
class FormatInverseKinematics:
    def __init__(self, action_dim, dv=0.005):
        self.action_dim = action_dim
        self.dv = dv

    def __call__(self, action):
        if self.action_dim == 1:
            dx = action[0] * self.dv
            dy = 0
            da = 0
        elif self.action_dim == 2:
            dx = action[0] * self.dv
            dy = action[1] * self.dv
            da = 0
        else:
            dx = action[0] * self.dv
            dy = action[1] * self.dv
            da = action[2] * 0.05

        f = 0.3
        action = [dx, dy, -0.002, da, f]

        return action
