import os
import numpy as np
import gymnasium as gym
from environment.gridworld import env as make_gridworld
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import stable_baselines3.common.env_checker as env_checker
from cnn.utils import preprocess_observation


# NOTE: This code doesn't work. PPO is deprecated and implemented in another repository.
class ScoutGymEnv(gym.Env):
    """
    A single-agent Gym wrapper around the PettingZoo gridworld for the scout role.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, novice: bool):
        super().__init__()
        # underlying PettingZoo AEC environment
        self.raw_env = make_gridworld(env_wrappers=[], novice=novice)
        self.scout_agent = self.raw_env.possible_agents[0]
        # Preprocess a sample observation to infer feature shapes
        self.raw_env.reset()
        obs, _, _, _, _ = self.raw_env.last()
        raw_obs_dict = obs
        proc = preprocess_observation({
            k: v if isinstance(v, int) else v.tolist()
            for k, v in raw_obs_dict.items()
        })
        vc = proc["viewcone"].squeeze(0).numpy()       # shape (8,7,5)
        vf = proc["vector_features"].squeeze(0).numpy()# shape (8,)
        flat_dim = vc.size + vf.size
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"),
            shape=(flat_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.raw_env.action_space(self.scout_agent).n)

    def reset(self, seed=None, options=None):
        # Reset the AEC environment
        try:
            self.raw_env.reset(seed=seed)
        except TypeError:
            # if seed not supported, call without seed
            self.raw_env.reset()
        # Fetch the initial observation via last()
        raw_obs, _, termination, truncation, info = self.raw_env.last()
        proc = preprocess_observation({
            k: v if isinstance(v, int) else v.tolist()
            for k, v in raw_obs.items()
        })
        vc = proc["viewcone"].squeeze(0).numpy().flatten()
        vf = proc["vector_features"].squeeze(0).numpy().flatten()
        flat_obs = np.concatenate((vc, vf), axis=0)
        # As per Gym API, reset_infos can be empty dict
        return flat_obs, {}

    def step(self, action):
        flat_obs = None
        reward = 0.0
        done = False
        info = {}

        # Step all agents in turn; capture scout's transition when it occurs
        for agent in self.raw_env.agent_iter():
            obs, rew, term, trunc, inf = self.raw_env.last()
            if agent == self.scout_agent and not (term or trunc):
                # Apply scout's chosen action
                self.raw_env.step(action)
                # Immediately fetch scout's outcome
                raw_obs, reward, term, trunc, info = self.raw_env.last()
                proc = preprocess_observation({
                    k: v if isinstance(v, int) else v.tolist()
                    for k, v in raw_obs.items()
                })
                vc = proc["viewcone"].squeeze(0).numpy().flatten()
                vf = proc["vector_features"].squeeze(0).numpy().flatten()
                flat_obs = np.concatenate((vc, vf), axis=0)
                done = term or trunc
            elif term or trunc:
                # If this agent is done, pass None (no-op)
                self.raw_env.step(None)
            else:
                # Non-scout agents take a random valid action
                rand_action = self.raw_env.action_space(agent).sample()
                self.raw_env.step(rand_action)

        # Return the scout's observation, reward, and done flag
        return flat_obs, reward, done, False, {}


class PPOManager:
    """PPO agent using Stable-Baselines3"""

    CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")
    CHECKPOINT_FILE = "rl_latest.pt"

    def __init__(self, lr: float = 1e-4, gamma: float = 0.99):
        # make sure checkpoint dir exists
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        # wrap the PettingZoo AEC env as a Gym Env for the scout
        self.env = DummyVecEnv([lambda: ScoutGymEnv(novice=(os.getenv("TEAM_TRACK")=="novice"))])
        # set up PPO with tensorboard logging
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=lr,
            gamma=gamma,
            tensorboard_log=self.CHECKPOINT_DIR,
            verbose=1,
        )
        ckpt_path = os.path.join(self.CHECKPOINT_DIR, self.CHECKPOINT_FILE)
        if os.path.isfile(ckpt_path):
            try:
                self.model = PPO.load(ckpt_path, env=self.env)
                print(f"[RLManager] Loaded PPO checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"[RLManager] Warning: could not load PPO checkpoint from {ckpt_path}: {e}")

    def train(self, total_timesteps: int):
        checkpoint_callback = CheckpointCallback(
            save_freq=10_000, save_path=self.CHECKPOINT_DIR, name_prefix="ppo_model"
        )
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            tb_log_name="PPO",
        )

        # save final model
        self.model.save(os.path.join(self.CHECKPOINT_DIR, self.CHECKPOINT_FILE))