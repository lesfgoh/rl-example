from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

from cnn.utils import preprocess_observation
from cnn.model import PolicyNet
from memory.frame_stack import FrameStack
from checkpoint.checkpoint import save_checkpoint, load_checkpoint
from buffer.episode_buffer import EpisodeBuffer


class ReinforceAgent:
    """
    On-policy REINFORCE agent supporting separate scout and guard policies.
    Guards use individual buffers but share a policy.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        scout_gamma: float = 0.9,
        guard_gamma: float = 0.7,
        history_len: int = 4,
        checkpoint_dir: str | None = None,
        entropy_beta: float = 0.01,
        checkpoint_save_interval: int = 100,
        histogram_log_interval: int = 500,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history_len = history_len
        self.guard_gamma = guard_gamma
        self.entropy_beta = entropy_beta
        self.checkpoint_save_interval = checkpoint_save_interval
        self.histogram_log_interval = histogram_log_interval

        # Networks
        self.scout_net = PolicyNet(n_actions=4, history_len=history_len).to(self.device)
        self.guard_net = PolicyNet(n_actions=4, history_len=history_len).to(
            self.device
        )  # Shared policy for guards
        self.scout_net.train()
        self.guard_net.train()

        # Optimizers
        self.scout_optim = torch.optim.Adam(self.scout_net.parameters(), lr=lr)
        self.guard_optim = torch.optim.Adam(self.guard_net.parameters(), lr=lr)

        # Scout-specific buffers
        self.scout_buffer = EpisodeBuffer(gamma=scout_gamma)
        self.scout_frame_stack = FrameStack(history_len=history_len, device=self.device)

        # Guard-specific buffers (dictionaries keyed by agent_id)
        self.guard_buffers: dict[str, EpisodeBuffer] = {}
        self.guard_frame_stacks: dict[str, FrameStack] = {}

        # Checkpoint config
        # uses relative loading, ckpt file relative to ckpt_dir
        self.ckpt_dir = "./checkpoints"
        self.policy_dir = "./checkpoints/policies"
        self.policy_file = f"latest-history-len-{self.history_len}.pt"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.policy_dir, exist_ok=True)

        # Logging
        self.total_steps = 0  # Number of episodes
        self.scout_writer = SummaryWriter(
            log_dir=os.path.join(self.ckpt_dir, "weights_scout")
        )
        self.guard_writer = SummaryWriter(
            log_dir=os.path.join(self.ckpt_dir, "weights_guard")
        )

        try:
            self._load_checkpoint()
        except FileNotFoundError:
            print("[ReinforceAgent] No existing checkpoint, starting fresh.")
        except Exception as e:
            print(f"[ReinforceAgent] Error loading checkpoint: {e}. Starting fresh.")

    def select_action(
        self, observation: Dict[str, Any]
    ) -> int:  # observation type updated
        proc = preprocess_observation(observation)
        vc_t, vf_t = proc["viewcone"], proc["vector_features"]

        agent_id = observation["agent_id"]
        is_scout = observation.get("scout")

        current_frame_stack: FrameStack
        current_buffer: EpisodeBuffer
        net: PolicyNet

        if is_scout:
            current_frame_stack = self.scout_frame_stack
            current_buffer = self.scout_buffer
            net = self.scout_net
        else:  # Guard
            current_frame_stack = self.guard_frame_stacks.setdefault(
                agent_id, FrameStack(self.history_len, self.device)
            )
            current_buffer = self.guard_buffers.setdefault(
                agent_id, EpisodeBuffer(self.guard_gamma)
            )
            net = self.guard_net

        current_frame_stack.append(vc_t, vf_t)
        vc, vf = current_frame_stack.get()

        logits = net(vc, vf)
        probs = F.softmax(logits, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        entropy = dist.entropy()

        current_buffer.store_action_data(dist.log_prob(action), entropy)

        return int(action.item())

    def record_reward(
        self, reward: float, observation: Dict[str, Any]
    ) -> None:  # observation type updated
        agent_id = observation["agent_id"]  # Expect agent_id
        is_scout = observation.get("scout", 0) == 1

        current_buffer: EpisodeBuffer
        if is_scout:
            current_buffer = self.scout_buffer
        else:  # Guard
            current_buffer = self.guard_buffers.setdefault(
                agent_id, EpisodeBuffer(self.guard_gamma)
            )  # Ensure buffer exists

        current_buffer.record_reward(reward, observation)

    def update(self) -> None:
        # Scout update (remains largely the same)
        if self.scout_buffer.log_probs:
            if len(self.scout_buffer.rewards) > len(self.scout_buffer.log_probs):
                del self.scout_buffer.rewards[0]

            returns, _, entropies = self.scout_buffer.compute_returns()
            returns = returns.to(self.device)

            policy_losses = [
                -lp * G for lp, G in zip(self.scout_buffer.log_probs, returns)
            ]
            entropy_bonuses = [H for H in entropies]

            policy_loss = torch.stack(policy_losses).sum()
            entropy_sum = torch.stack(entropy_bonuses).sum()
            loss = policy_loss - self.entropy_beta * entropy_sum

            self.scout_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scout_net.parameters(), max_norm=1.0)
            self.scout_optim.step()

            avg_entropy_scout = (
                entropy_sum / len(entropies) if entropies else torch.tensor(0.0)
            ).item()
            self.scout_writer.add_scalar(
                "scout/episode_entropy", avg_entropy_scout, self.total_steps
            )
            self.scout_writer.add_scalar(
                "scout/policy_loss", policy_loss.item(), self.total_steps
            )
            self.scout_writer.add_scalar(
                "scout/total_loss", loss.item(), self.total_steps
            )
            if self.total_steps % self.histogram_log_interval == 0:
                for name, param in self.scout_net.named_parameters():
                    self.scout_writer.add_histogram(
                        f"scout/{name}", param, self.total_steps
                    )
            self.scout_buffer.clear()

        # Guard update (aggregated from individual buffers)
        all_guard_losses = []
        total_guard_policy_loss_val = 0.0
        total_guard_entropy_sum_val = 0.0
        num_guard_experiences_in_update = 0

        for agent_id, buffer in self.guard_buffers.items():
            if buffer.log_probs:  # Check if buffer has data
                if len(buffer.rewards) > len(buffer.log_probs):  # Basic alignment check
                    del buffer.rewards[0]

                returns, _, entropies = buffer.compute_returns()
                returns = returns.to(self.device)

                agent_policy_losses = [
                    -lp * G for lp, G in zip(buffer.log_probs, returns)
                ]
                agent_entropy_bonuses = [H for H in entropies]

                if not agent_policy_losses:
                    continue  # Skip if, after alignment, no data

                policy_loss_agent = torch.stack(agent_policy_losses).sum()
                entropy_sum_agent = torch.stack(agent_entropy_bonuses).sum()

                loss_agent = policy_loss_agent - self.entropy_beta * entropy_sum_agent
                all_guard_losses.append(loss_agent)

                total_guard_policy_loss_val += policy_loss_agent.item()
                total_guard_entropy_sum_val += entropy_sum_agent.item()
                num_guard_experiences_in_update += len(entropies)

        if all_guard_losses:
            # Sum losses from all guard agents that had experiences
            aggregated_guard_loss = torch.stack(all_guard_losses).sum()

            self.guard_optim.zero_grad()
            aggregated_guard_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.guard_net.parameters(), max_norm=1.0)
            self.guard_optim.step()

            avg_entropy_guard = (
                total_guard_entropy_sum_val / num_guard_experiences_in_update
                if num_guard_experiences_in_update > 0
                else 0.0
            )
            self.guard_writer.add_scalar(
                "guard/episode_entropy", avg_entropy_guard, self.total_steps
            )
            self.guard_writer.add_scalar(
                "guard/policy_loss", total_guard_policy_loss_val, self.total_steps
            )
            self.guard_writer.add_scalar(
                "guard/total_loss", aggregated_guard_loss.item(), self.total_steps
            )
            if self.total_steps % self.histogram_log_interval == 0:
                for name, param in self.guard_net.named_parameters():
                    self.guard_writer.add_histogram(
                        f"guard/{name}", param, self.total_steps
                    )

        # Clear all individual guard buffers after update
        for buffer in self.guard_buffers.values():
            buffer.clear()

        self.total_steps += 1

        self.scout_writer.flush()
        self.guard_writer.flush()

        if self.total_steps % self.checkpoint_save_interval == 0:
            save_checkpoint(  # Save updated shared guard_net
                policy_dir=self.policy_dir,
                scout_net=self.scout_net,
                scout_optim=self.scout_optim,
                guard_net=self.guard_net,
                guard_optim=self.guard_optim,
                filename=self.policy_file,
            )
            if (
                self.checkpoint_save_interval > 1
            ):  # Avoid spamming for every episode if interval is 1
                print(
                    f"[ReinforceAgent] Checkpoint saved at episode {self.total_steps}"
                )

    def _load_checkpoint(self) -> None:
        data = load_checkpoint(
            self.policy_dir, self.policy_file, map_location=self.device
        )
        self.scout_net.load_state_dict(data["scout_state_dict"])
        self.scout_optim.load_state_dict(data["scout_optim_state"])
        self.guard_net.load_state_dict(data["guard_state_dict"])  # Shared guard net
        self.guard_optim.load_state_dict(data["guard_optim_state"])
        # self.total_steps might need to be loaded if saved in checkpoint

    def reset(self) -> None:
        self.scout_frame_stack.reset()
        # self.scout_buffer.clear() # Clearing is done in update now

        # Clear all guard-specific frame stacks and buffers
        for stack in self.guard_frame_stacks.values():
            stack.reset()
        # for buffer in self.guard_buffers.values(): # Clearing is done in update now
        #     buffer.clear()

        # It's safer to clear the dicts themselves to remove old agent_ids
        # if agents can change between episodes. Buffers are cleared in update.
        # FrameStacks need to be reset for the new episode.
        # New buffers/stacks will be created on demand via setdefault.
        # However, to prevent unbounded growth if agent IDs are very dynamic and short-lived:
        self.guard_frame_stacks.clear()  # Clears for new episode
        self.guard_buffers.clear()  # Clears for new episode, new ones made by setdefault
