from stable_baselines3 import PPO
from pokebot_env import (
    PokeBotEnv,
)

from genericpath import exists
from pathlib import Path
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import RecurrentPPO

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

drive = "/content/drive/MyDrive/Pokemon/"
goal = "oaks_parcel"
session_path = "/Sessions/"
tensorboard_path = "/Tensorboard/"
ep_length = 3200


def _create_env():
    def func():
        return Monitor(PokeBotEnv(True, step_limit=ep_length))

    return func


def _environments(count):
    return list(map(lambda _: _create_env(), range(count)))


def train():
    sess_path = Path(f"{session_path}/{goal}")
    environment_count = 1
    env = DummyVecEnv(_environments(environment_count))
    

    nsteps = ep_length // 64
    file_name = f"{session_path}/oaks_parcel/train_50000_steps"

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = RecurrentPPO.load(file_name, env=env, tensorboard_log=tensorboard_path)
        model.n_steps = ep_length
        model.n_envs = 1
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = 1
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            # "MultiInputLstmPolicy",
            env,
            verbose=1,
            n_steps=nsteps,
            batch_size=nsteps,
            n_epochs=1,
            tensorboard_log=tensorboard_path,
            gamma=0.99,
        )

    while True:
        model.learn(
            total_timesteps=ep_length * 100,
            tb_log_name=f"{goal}",
            reset_num_timesteps=True,
            # progress_bar=True,
        )

        model.save(f"{drive}/model/{goal}")

