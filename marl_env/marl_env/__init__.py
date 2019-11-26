from gym.envs.registration import register

register(
    id='marl-farm-v0',
    entry_point='marl_env.envs:FarmMARL',
)