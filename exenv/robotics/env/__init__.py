from gym.envs.registration import register

register(
    id='LiftBlockWithKukaInverseKinematics-v0',
    entry_point='exenv.robotics.env.gym_env:LiftBlockWithKukaInverseKinematics'
)
