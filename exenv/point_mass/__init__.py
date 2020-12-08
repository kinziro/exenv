from gym.envs.registration import register

register(
    id='PointMass-v0',
    entry_point='exenv.point_mass.point_mass_env:PointMassEnv'
)
register(
    id='PointMassMultiTransition-v0',
    entry_point='exenv.point_mass.point_mass_env_multi_transition:PointMassEnv'
)
register(
    id='PointMassAddEnvId-v0',
    entry_point='exenv.point_mass.point_mass_env_add_env_id:PointMassEnv'
)
register(
    id='PointMassLineTrace-v0',
    entry_point='exenv.point_mass.point_mass_line_trace_env:PointMassEnv'
)
register(
    id='PointMassObsError-v0',
    entry_point='exenv.point_mass.point_mass_env_obs_error:PointMassEnv'
)