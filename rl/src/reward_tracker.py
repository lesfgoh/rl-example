import numpy as np
from torch.utils.tensorboard import SummaryWriter
from agent_utils import get_agent_info
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from pettingzoo.utils.env import AECEnv
    from rl_manager import RLManager


class RewardTracker:
    """
    Tracks and logs rewards during training, providing detailed reporting.
    """

    def __init__(self, log_dir="runs/train_rl", detailed_rewards=False):
        """
        Initialize the reward tracker

        Args:
            log_dir: Directory for TensorBoard logs
            detailed_rewards: Whether to print detailed reward information
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.detailed_rewards = detailed_rewards
        self.episode_returns = []

    def record_termination_rewards(
        self, env: "AECEnv", agent_observations: Dict[str, Any], manager: "RLManager"
    ):
        """
        Record termination rewards from environment

        Args:
            env: The environment object with rewards dictionary
            agent_observations: Dictionary mapping agents to their observations
            manager: RL manager to record rewards

        Returns:
            bool: True if termination rewards were processed
        """
        if not hasattr(env, "rewards") or not env.rewards:
            return False

        if self.detailed_rewards:
            print(f"\n*** TERMINATION EVENT - Environment rewards: {env.rewards} ***")

        for env_agent_raw, env_reward in env.rewards.items():
            if env_reward == 0:
                continue

            # Use utility to get mapped ID for manager and scout status
            # We don't strictly need raw_observation here if env.scout is reliable for termination phase
            mapped_agent_id_for_manager, _, _ = get_agent_info(env_agent_raw, env)

            agent_obs_for_local_use = agent_observations.get(env_agent_raw)

            if self.detailed_rewards:
                is_scout_for_log = "scout" in mapped_agent_id_for_manager
                print(
                    f"DEBUG RewardTracker: Raw env_agent='{env_agent_raw}', Mapped to='{mapped_agent_id_for_manager}', env.scout='{getattr(env, 'scout', 'N/A')}', IsScoutLogic={is_scout_for_log}"
                )
                print(
                    f"DEBUG RewardTracker: agent_observations keys: {list(agent_observations.keys())}"
                )
                if agent_obs_for_local_use is None:
                    print(
                        f"DEBUG RewardTracker: CRITICAL - Observation for raw_id '{env_agent_raw}' not found in agent_observations! This means it likely had no prior non-terminal steps."
                    )
                print(
                    f"Recording termination reward {env_reward:.1f} for MAPPED_ID '{mapped_agent_id_for_manager}' (from raw '{env_agent_raw}') by updating last reward."
                )

            manager.update_agent_terminal_reward(
                mapped_agent_id_for_manager, env_reward
            )

        return True

    def record_step_reward(self, agent, reward, observation, manager):
        """
        Record regular step rewards

        Args:
            agent: Current agent
            reward: Reward value
            prev_observation: Previous observation to associate with the reward
            manager: RL manager to record rewards
        """
        # In detailed mode, only report non-zero rewards to reduce noise
        # But always send the reward to the manager
        if self.detailed_rewards and reward != 0:
            print(f"Recording reward {reward:.1f} for agent {agent}")

        # Always record the reward regardless of value
        manager.record_reward(reward, observation)

    def end_episode(self, round_idx, num_rounds, manager):
        """
        Process the end of an episode, collecting and logging rewards

        Args:
            round_idx: Current episode index
            num_rounds: Total number of episodes
            manager: RL manager with reward information

        Returns:
            tuple: Scout and guard returns for this episode
        """
        agent = manager.agent
        scout_buffer = getattr(agent, "scout_buffer", None)

        scout_returns = 0.0
        guards_total_returns = 0.0

        if scout_buffer and hasattr(scout_buffer, "total_reward"):
            scout_returns = scout_buffer.total_reward

        if hasattr(agent, "guard_buffers") and isinstance(agent.guard_buffers, dict):
            for _guard_id, guard_episode_buffer in agent.guard_buffers.items():
                if hasattr(guard_episode_buffer, "total_reward"):
                    guards_total_returns += guard_episode_buffer.total_reward

        self.episode_returns.append(scout_returns)
        self.writer.add_scalar("scout/return", scout_returns, round_idx)
        self.writer.add_scalar(
            "guard/total_episode_return", guards_total_returns, round_idx
        )

        return scout_returns, guards_total_returns

    def print_summary(self):
        """
        Print a summary of all episodes

        Returns:
            float: Competition score
        """
        if not self.episode_returns:
            print("No episodes completed.")
            return 0.0

        avg_return = np.mean(self.episode_returns)
        competition_score = self.compute_score()

        print(f"Average return: {avg_return:.2f}")
        print(f"Competition score: {competition_score:.4f}")

        return competition_score

    def compute_score(self):
        """
        Compute the competition score

        Returns:
            float: Competition score (average reward per episode divided by 100)
        """
        if not self.episode_returns:
            return 0.0
        avg_reward = sum(self.episode_returns) / len(self.episode_returns)
        return avg_reward / 100

    def close(self):
        """
        Close the tracker and clean up resources
        """
        self.writer.close()
