#!/usr/bin/env python3
"""
Headless training module for faster training without visualization.
"""

import os
import argparse
import time
import numpy as np  # Import numpy
from environment.gridworld import env as make_gridworld
from rl_manager import RLManager
from reward_tracker import RewardTracker
from agent_utils import get_agent_info
from typing import Dict, Any


def eval_model(
    detailed_rewards: bool = False,
    history_len: int = 4,
):
    manager = RLManager(
        history_len=history_len,
        lr=2e-5,
        scout_gamma=0.75,
        guard_gamma=0.99,
        entropy_beta=0.02,
    )

    # Create environment without visualization
    env = make_gridworld(
        env_wrappers=[],
        novice=False,
        render_mode="human",
        debug=False,
    )

    while True:
        manager.reset()
        env.reset()

        for agent_raw_id in env.agent_iter():  # example: "player_0", "player_1"
            (
                current_obs_raw,
                current_reward_from_env,
                termination,
                truncation,
                info,
            ) = env.last()

            current_obs_raw["agent_id"] = agent_raw_id

            manager.record_reward(current_reward_from_env, current_obs_raw)

            if termination or truncation:
                break

            env.step(manager.rl(current_obs_raw))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL models")
    parser.add_argument(
        "--history-len", type=int, default=4, help="Model of history len to run"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable detailed reward output"
    )

    args = parser.parse_args()

    eval_model(
        history_len=args.history_len,
        detailed_rewards=args.verbose,
    )
