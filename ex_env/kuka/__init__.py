from gym.envs.registration import register

register(
    id='KukaEnvExperiment-v0',
    entry_point='ex_env.kuka.kukaGymEnv:KukaGymEnv'
)