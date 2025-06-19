#!/usr/bin/env python3
"""
Headless training module for faster training without visualization.
"""

import os
import argparse
import time
import numpy as np
from environment.gridworld import env as make_gridworld
from rl_manager import RLManager
from reward_tracker import RewardTracker
from agent_utils import get_agent_info
from typing import Dict, Any


def train_headless(
    num_rounds: int = 1000000000,
    detailed_rewards: bool = False,
    checkpoint_interval: int = 100,
    entropy_beta: float = 0.02,
    checkpoint_save_interval: int = 100,
    histogram_log_interval: int = 500,
    history_len: int = 4,
    debug_human: bool = True,
):
    """
    Run headless training without visualization for faster execution

    Args:
        num_rounds: Number of episodes to run
        detailed_rewards: Whether to print detailed reward information
        checkpoint_interval: How often to report progress (in episodes)
        entropy_beta: Coefficient for entropy regularization
        checkpoint_save_interval: How often to save model checkpoints (in episodes)
        histogram_log_interval: How often to log histograms to TensorBoard
    """

    novice = os.getenv("TEAM_TRACK", "") == "advanced"

    exploration_reward = 0.07
    no_reward_penalty = 0.05
    guard_shaping_closer = 0.07
    guard_shaping_further = 0.05

    reward_tracker = RewardTracker(detailed_rewards=detailed_rewards)

    manager = RLManager(
        history_len=history_len,
        lr=2e-5,
        scout_gamma=0.97,
        guard_gamma=0.99,
        entropy_beta=entropy_beta,
        checkpoint_save_interval=checkpoint_save_interval,
        histogram_log_interval=histogram_log_interval,
    )

    env = make_gridworld(
        env_wrappers=[],
        novice=False,
        render_mode="human" if debug_human else None,
        debug=False,
    )

    # Tracking variables
    start_time = time.time()
    report_interval_time = time.time()
    report_interval_steps = 0
    report_interval_scout_returns = 0
    report_interval_guard_returns = 0

    try:
        for round_idx in range(1, num_rounds + 1):
            manager.reset()
            env.reset()

            terminated_episode = False
            episode_steps = 0
            agent_observations: Dict[str, Any] = {}
            previous_distances_to_scout: Dict[
                str, float
            ] = {}  # Stores previous distances for guards for reward shaping
            agent_last_positions_this_episode: Dict[
                str, list
            ] = {} 

            for agent_raw_id in env.agent_iter():
                episode_steps += 1
                report_interval_steps += 1

                (
                    current_obs_raw,
                    current_reward_from_env,
                    termination,
                    truncation,
                    info,
                ) = env.last()

                # Test: Penalise for no rewards. Flag! Need to be careful with this.
                if current_reward_from_env == 0:
                    current_reward_from_env -= no_reward_penalty

                mapped_agent_id, scout_flag, _ = get_agent_info(
                    agent_raw_id, env, current_obs_raw
                )
                current_total_step_reward = float(current_reward_from_env)

                # ---- New Cell Exploration Bonus ----
                current_pos = None
                if current_obs_raw is not None:
                    current_pos = current_obs_raw.get("location")

                if current_pos is not None:
                    visited_positions_for_agent = agent_last_positions_this_episode.get(
                        mapped_agent_id, []
                    )

                    # Check if current_pos is in the list of visited arrays
                    is_new_position = not any(
                        np.array_equal(current_pos, visited_pos)
                        for visited_pos in visited_positions_for_agent
                    )

                    if is_new_position:
                        visited_positions_for_agent.append(
                            current_pos.copy()
                        )  # Store a copy
                        agent_last_positions_this_episode[mapped_agent_id] = (
                            visited_positions_for_agent
                        )
                        if detailed_rewards:
                            print(
                                f"DEBUG Agent {mapped_agent_id}: New cell! Pos: {current_pos}. ExpRew +{exploration_reward}"
                            )

                        current_total_step_reward += exploration_reward
                # ---- End New Cell Exploration Bonus ----

                # ---- Reward Shaping for Guards based on distance ----
                if (
                    not scout_flag
                    and info
                    and isinstance(info, dict)
                    and "distance" in info
                ):  # Check info is dict
                    current_distance = float(info["distance"])  # or info['manhattan']

                    prev_distance = previous_distances_to_scout.get(mapped_agent_id)
                    if prev_distance is not None:
                        if current_distance <= prev_distance:
                            if detailed_rewards:
                                print(
                                    f"DEBUG Guard {mapped_agent_id}: Closer! {prev_distance:.1f} -> {current_distance:.1f}. ShapRew +{guard_shaping_closer}"
                                )

                            current_total_step_reward += guard_shaping_closer * (
                                prev_distance - current_distance
                            )
                        # # penalize moving further away, but start without it
                        elif current_distance > prev_distance:
                            if detailed_rewards:
                                print(
                                    f"DEBUG Guard {mapped_agent_id}: Further! {prev_distance:.1f} -> {current_distance:.1f}. ShapRew -{guard_shaping_further}"
                                )
                            current_total_step_reward -= guard_shaping_further * (
                                current_distance - prev_distance
                            )

                    previous_distances_to_scout[mapped_agent_id] = current_distance
                # ---- End Reward Shaping ----

                current_observation_processed = None
                if current_obs_raw is not None:
                    current_observation_processed = current_obs_raw.copy()
                    current_observation_processed["agent_id"] = mapped_agent_id
                    current_observation_processed["scout"] = (
                        scout_flag  # Use scout_flag from utility
                    )

                if current_observation_processed is not None:
                    agent_observations[agent_raw_id] = current_observation_processed

                obs_for_reward_recording = agent_observations.get(agent_raw_id)

                if not terminated_episode:
                    if obs_for_reward_recording is not None:
                        reward_tracker.record_step_reward(
                            mapped_agent_id,
                            current_total_step_reward,
                            obs_for_reward_recording,
                            manager,
                        )

                if (termination or truncation) and not terminated_episode:
                    terminated_episode = reward_tracker.record_termination_rewards(
                        env, agent_observations, manager
                    )

                next_action_to_take = None
                if not (termination or truncation):
                    if current_observation_processed is not None:
                        next_action_to_take = manager.rl(current_observation_processed)

                env.step(next_action_to_take)

            scout_returns, guard_returns = reward_tracker.end_episode(
                round_idx, num_rounds, manager
            )

            # track returns
            report_interval_guard_returns += guard_returns
            report_interval_scout_returns += scout_returns

            manager.update()

            if round_idx % checkpoint_interval == 0:
                elapsed = time.time() - report_interval_time
                steps_per_sec = report_interval_steps / elapsed if elapsed > 0 else 0
                avg_guard_returns = report_interval_guard_returns / (
                    checkpoint_interval if checkpoint_interval > 0 else 1
                )
                avg_scout_returns = report_interval_scout_returns / (
                    checkpoint_interval if checkpoint_interval > 0 else 1
                )
                print(
                    f"Progress: {round_idx}/{num_rounds} episodes, {report_interval_steps} steps ({steps_per_sec:.1f} steps/sec), avg scout {avg_scout_returns:.2f}, avg guard {avg_guard_returns:.2f}, elapsed {elapsed:.2f}s"
                )
                report_interval_time = time.time()
                report_interval_steps = 0
                report_interval_guard_returns = 0
                report_interval_scout_returns = 0

    finally:
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.2f} seconds")

        reward_tracker.close()
        env.close()

    reward_tracker.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent in headless mode (")
    parser.add_argument(
        "--episodes", type=int, default=1000000000, help="Number of episodes to run"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable detailed reward output"
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=100,
        help="How often to report progress (in episodes)",
    )
    parser.add_argument(
        "--entropy-beta",
        type=float,
        default=0.02,
        help="Coefficient for entropy regularization term",
    )
    parser.add_argument(
        "--checkpoint-save-interval",
        type=int,
        default=100,
        help="How often to save model checkpoints (in episodes)",
    )
    parser.add_argument(
        "--histogram-log-interval",
        type=int,
        default=500,
        help="How often to log histograms to TensorBoard (in episodes)",
    )

    args = parser.parse_args()

    train_headless(
        num_rounds=args.episodes,
        detailed_rewards=args.verbose,
        checkpoint_interval=args.report_interval,
        entropy_beta=args.entropy_beta,
        checkpoint_save_interval=args.checkpoint_save_interval,
        histogram_log_interval=args.histogram_log_interval,
    )
