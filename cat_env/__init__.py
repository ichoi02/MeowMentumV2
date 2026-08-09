from gymnasium.envs.registration import register

register(
    id="Cat-v0",
    entry_point="cat_env.cat_env:CatEnv",
    max_episode_steps=50,
    kwargs={"model_path": "model/cat.xml"},
)

register(
    id="CatNoTail-v0",
    entry_point="cat_env.cat_env:CatEnv",
    max_episode_steps=50,
    kwargs={"model_path": "model/cat_notail.xml"},
)