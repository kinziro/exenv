from gym.envs.registration import register

register(
    id='PointMass-v0',
    entry_point='exenv.point_mass.point_mass_env:PointMassEnv'
)