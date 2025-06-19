from algorithms.reinforce import ReinforceAgent


class RLManager:
    def __init__(
        self,
        algo="reinforce",
        entropy_beta: float = 0.02,
        checkpoint_save_interval: int = 100,
        histogram_log_interval: int = 500,
        **kwargs,
    ):
        if algo == "reinforce":
            self.agent = ReinforceAgent(
                entropy_beta=entropy_beta,
                checkpoint_save_interval=checkpoint_save_interval,
                histogram_log_interval=histogram_log_interval,
                **kwargs,
            )
        # Expose properties for server
        self.history_len = self.agent.history_len
        self.device = self.agent.device

    def rl(self, obs):
        return self.agent.select_action(obs)

    def record_reward(self, r, obs):
        return self.agent.record_reward(r, obs)

    def update_agent_terminal_reward(self, agent_type: str, terminal_reward: float):
        """Update the last reward of an agent's buffer with the terminal reward."""
        buffer_to_update = None
        # Assuming ReinforceAgent has scout_buffer and guard_buffers (dict)

        # Check if the agent_type indicates a scout or a guard
        # This logic depends on your agent naming convention (e.g., "scout_0", "guard_0")
        if "scout" in agent_type.lower():  # Or a more robust check
            if hasattr(self.agent, "scout_buffer"):
                buffer_to_update = self.agent.scout_buffer
            else:
                print(f"Warning: scout_buffer not found on agent for {agent_type}")
        elif "guard" in agent_type.lower():  # Or a more robust check
            if hasattr(self.agent, "guard_buffers") and isinstance(
                self.agent.guard_buffers, dict
            ):
                buffer_to_update = self.agent.guard_buffers.get(agent_type)
                if buffer_to_update is None:
                    print(
                        f"Warning: No buffer found for guard agent_id {agent_type}. Might be normal if guard had no actions."
                    )
            else:
                print(
                    f"Warning: guard_buffers dictionary not found on agent for {agent_type}"
                )
        else:
            print(
                f"Warning: Unknown agent type for agent_id {agent_type} in update_agent_terminal_reward"
            )

        if buffer_to_update and hasattr(buffer_to_update, "update_last_reward"):
            buffer_to_update.update_last_reward(terminal_reward)
        elif buffer_to_update:
            print(
                f"Warning: Buffer for {agent_type} does not have update_last_reward method."
            )
        # If buffer_to_update is None and it was a guard, it might be that the guard took no actions
        # and thus its buffer was never created by setdefault. This might be acceptable.

    def update(self) -> None:
        """
        Perform end-of-episode policy update.
        """
        self.agent.update()

    def reset(self) -> None:
        """
        Clear episode buffers and reinitialize frame history before a new episode.
        """
        self.agent.reset()
