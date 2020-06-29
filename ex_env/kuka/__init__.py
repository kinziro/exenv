from gym.envs.registration import register

register(
    id='KukaEnvReaching-v0',
    entry_point='ex_env.kuka.kukaEnvReaching:KukaGymEnv'
)