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

import os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

drive = "/content/drive/MyDrive/Pokemon/"
goal = "oaks_parcel"
session_path = "/Sessions/"
tensorboard_path = "/Tensorboard/"

step_limit = 5000
ep_length = 5000

def _delete_directory(path):
    
    if os.path.exists(path):

        import shutil
        shutil.rmtree(path)

_delete_directory("logs")
_delete_directory("actions")

num_cpus = os.cpu_count()


def _create_env(id):
    def func():
        return Monitor(PokeBotEnv(True, step_limit=step_limit, id=id))

    return func


def _environments(count):
    return list(map(lambda i: _create_env(i), range(count)))


def train():
    sess_path = Path(f"{session_path}/{goal}")
    environment_count = 1
    env = SubprocVecEnv(_environments(environment_count))
    

    nsteps = ep_length
    file_name = f"{session_path}/oaks_parcel/train_50000_steps"
    model_path = f"{drive}/model/{goal}"

    if exists(model_path + ".zip"):
        model = PPO.load(f"{model_path}.zip")
    else:
        model = PPO(
            "MultiInputPolicy",
            # "MultiInputLstmPolicy",
            env,
            verbose=1,
            n_steps=nsteps,
            batch_size=nsteps,
            n_epochs=1,
            gamma=0.99,
            tensorboard_log=tensorboard_path,
        )

    while True:
        try:
            directory = os.path.dirname(os.path.abspath(__file__))+"/"
            os.remove(directory+'states/inprogress.state')
        except:
            pass
        
        model.learn(
            total_timesteps=ep_length * 20,
            tb_log_name=f"{goal}",
            reset_num_timesteps=True,
            progress_bar=False,
        )

        model.save(f"{drive}/model/{goal}")


if __name__ == "__main__":
    train()