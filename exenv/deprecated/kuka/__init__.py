from gym.envs.registration import register

register(
    id='KukaEnvExperiment-v0',
    entry_point='exenv.deprecated.kuka.kukaGymEnv:KukaGymEnv'
)
