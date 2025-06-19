import torch
from typing import List, Dict, Any, Tuple
import warnings

class EpisodeBuffer:
    """
    Stores log-probabilities, rewards, and entropies for a single agent
    over an episode, and computes discounted returns.
    """

    def __init__(self, gamma: float = 0.99):
        """
        :param gamma: discount factor for returns
        """
        self.gamma = gamma
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.entropies: List[torch.Tensor] = []  # New: store entropies
        # Add detailed reward tracking with context
        self.reward_details: List[Dict[str, Any]] = []
        self.total_reward: float = 0.0
        # Debug tracking
        self.debug_info: Dict[str, Any] = {
            "log_probs_count": 0,
            "rewards_count": 0,
            "entropies_count": 0  # New: track entropies count
        }

    def append(self, log_prob: torch.Tensor, reward: float) -> None:
        """
        Add one timestep's log-probability and scalar reward.
        DEPRECATED: Use store_action_data and record_reward separately
        """
        # This method is deprecated, but let's add a placeholder for entropy
        # to avoid errors if it's somehow still called.
        # A better approach would be to remove this method entirely.
        warnings.warn("EpisodeBuffer.append() is deprecated. Use store_action_data and record_reward.", DeprecationWarning)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.entropies.append(torch.tensor(0.0)) # Placeholder entropy
        self.total_reward += reward
        self.debug_info["log_probs_count"] += 1
        self.debug_info["rewards_count"] += 1
        self.debug_info["entropies_count"] += 1
        
        # Add minimal reward detail when called through append
        self.reward_details.append({
            "reward": reward,
            "is_scout": None,  # Unknown agent type in this context
            "step_count": len(self.rewards)
        })

    def store_action_data(self, log_prob: torch.Tensor, entropy: torch.Tensor) -> None: # Renamed and updated
        """Store the log probability and entropy from an action."""
        self.log_probs.append(log_prob)
        self.entropies.append(entropy) # Store entropy
        self.debug_info["log_probs_count"] += 1
        self.debug_info["entropies_count"] += 1

    def record_reward(self, reward: float, observation: Dict[str, Any]) -> None:
        """
        Record a reward with associated observation context
        """
        # Don't append to rewards list if just using the record_reward path directly
        self.rewards.append(reward)
        self.total_reward += reward
        self.debug_info["rewards_count"] += 1
        
        # Store detailed information about this reward
        self.reward_details.append({
            "reward": reward,
            "is_scout": observation.get("scout", 0),
            "step_count": len(self.rewards) 
        })

    def update_last_reward(self, terminal_reward: float) -> None:
        """Update the last recorded reward, typically with a terminal reward."""
        if not self.rewards:
            warnings.warn("EpisodeBuffer.update_last_reward() called on an empty rewards list. "
                          "This may indicate a logic error where a terminal reward is applied "
                          "without prior steps/rewards.", UserWarning)
            return

        # Adjust total_reward: subtract old last reward, add new terminal reward
        self.total_reward -= self.rewards[-1]
        self.total_reward += terminal_reward
        
        # Update the reward value itself
        self.rewards[-1] = terminal_reward
        
        # Update the corresponding entry in reward_details
        if self.reward_details:
            self.reward_details[-1]["reward"] = terminal_reward

    def compute_returns(self) -> Tuple[torch.Tensor, List[Dict[str, Any]], List[torch.Tensor]]: # Updated return type
        """
        Return a tensor of discounted returns, detailed reward information,
        and a list of entropies.
        """
        # Handle mismatch between log_probs, rewards, and entropies
        min_len = len(self.rewards)
        if len(self.log_probs) != min_len or len(self.entropies) != min_len:
            warnings.warn(f"ERROR (EpisodeBuffer): Unexpected mismatch between log_probs ({len(self.log_probs)}), "
                          f"rewards ({len(self.rewards)}), and entropies ({len(self.entropies)}) "
                          f"when compute_returns() was called. "
                          f"Truncating to min_len={min_len} based on rewards list length.", UserWarning)
            
            # Truncate all lists to the length of the rewards list,
            # as rewards are fundamental for return calculation.
            # This assumes rewards list is the "source of truth" for episode length.
            # If rewards is longer, this implies missing log_probs/entropies, which is an issue.
            self.log_probs = self.log_probs[:min_len]
            self.entropies = self.entropies[:min_len]
            # If rewards is shorter, other lists have already been implicitly truncated by min_len logic.

        returns: List[float] = []
        R = 0.0
        # accumulate from the end
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        # normalize for stability
        if returns_tensor.numel() > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-6)
        return returns_tensor, self.reward_details, self.entropies # Return entropies

    def get_reward_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the rewards in this episode
        """
        if not self.reward_details:
            return {
                "total": 0.0, 
                "count": 0, 
                "rewards": [],
                "debug": self.debug_info
            }
            
        return {
            "total": self.total_reward,
            "count": len(self.rewards),
            "rewards": self.reward_details,
            "debug": self.debug_info
        }

    def clear(self) -> None:
        """
        Reset buffer for the next episode.
        """
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()  # New: clear entropies
        self.reward_details.clear()
        self.total_reward = 0.0
        # Reset debug tracking
        self.debug_info = {
            "log_probs_count": 0,
            "rewards_count": 0,
            "entropies_count": 0 # New: reset entropies count
        }