from gym.envs.registration import register

register(
    id='LiftBlockWithKukaInverseKinematics-v0',
    entry_point='exenv.robotics.env.gym_env:LiftBlockWithKukaInverseKinematics'
)

register(
    id='LiftBlockWithUr5InverseKinematics-v0',
    entry_point='exenv.robotics.env.gym_env:LiftBlockWithUr5InverseKinematics'
)
