import json
import os
from torch.utils.tensorboard import SummaryWriter
import requests
from dotenv import load_dotenv
from environment import gridworld
import numpy as np
from pprint import pprint


def score(episode_returns):
    """
    Compute the competition score:
    average reward per episode divided by 100.
    """
    if not episode_returns:
        return 0.0
    avg_reward = sum(episode_returns) / len(episode_returns)
    return avg_reward / 100


load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

NUM_ROUNDS = 500000000


def main(novice: bool):
    writer = SummaryWriter(log_dir="runs/test_rl")  # create a log directory
    env = gridworld.env(env_wrappers=[], render_mode="human", novice=novice)
    # be the agent at index 0
    _agent = env.possible_agents[0]
    episode_returns = []
    total = 0

    for round_idx in range(NUM_ROUNDS):
        # start a new episode
        env.reset()
        scout_rewards, guard_rewards = 0, 0
        _ = requests.post("http://localhost:5004/reset")
        for agent in enumerate(env.agent_iter()):
            observation, reward, termination, truncation, agent_info = env.last()
            # pprint(env.last())
            observation = {
                k: v if type(v) is int else v.tolist() for k, v in observation.items()
            }
            # print(agent_info)

            # print(f"Observation: {observation}")
            # print(f"Reward: {reward}")
            # handle dead agents
            if termination or truncation:
                action = None
            else:
                # accumulate total reward for our scout
                if observation["scout"] == 1:
                    # print(reward)
                    total += reward
                    payload = {
                        "instances": [
                            {
                                "observation": observation,
                                # "reward": reward
                            }
                        ]
                    }
                    response = requests.post(
                        "http://localhost:5004/rl",
                        json=payload,
                    )
                    predictions = response.json()["predictions"]
                    action = int(predictions[0]["action"])
                else:
                    action = env.action_space(agent).sample()

            env.step(action)
            if observation.get("scout"):
                scout_rewards += reward
            else:
                guard_rewards += reward

        if round_idx == 0:
            round_idx += 1
        print(scout_rewards)
        print(guard_rewards)

    env.close()
    # show episode returns and moving average
    print("Episode returns:", episode_returns)
    window = min(50, len(episode_returns))
    if window > 1:
        mov_avg = np.convolve(episode_returns, np.ones(window) / window, mode="valid")
        print(f"{window}-episode moving average:", mov_avg.tolist())

    avg_return = sum(episode_returns) / len(episode_returns)
    print(f"Average return per episode: {avg_return:.2f}")
    print(f"Competition score: {score(episode_returns):.4f}")
    writer.close()


if __name__ == "__main__":
    main(TEAM_TRACK == "novice")
