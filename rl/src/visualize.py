#!/usr/bin/env python3
"""
Visualize the RL training process with full debugging information.
This module provides interactive visualization with step-by-step control.
"""

import os
import argparse
from environment.gridworld import env as make_gridworld
from rl_manager import RLManager
from visualization import VisualizationHandler
from reward_tracker import RewardTracker


def visualize(num_rounds: int = 5000, detailed_rewards: bool = True):
    """
    Run environment visualization with debugging controls
    
    Args:
        num_rounds: Maximum number of episodes to run
        detailed_rewards: Whether to print detailed reward information
    """
    # Create visualization handler (always in "human" mode)
    viz = VisualizationHandler(render_mode="human")
    
    # Set up environment parameters
    novice = (os.getenv("TEAM_TRACK", "") == "novice")
    
    # Initialize reward tracker with detailed information
    reward_tracker = RewardTracker(detailed_rewards=detailed_rewards)
    
    # Initialize RL manager
    manager = RLManager(history_len=4, lr=1e-3, gamma=0.5)
    
    # Create environment with human rendering and debug enabled
    env = make_gridworld(env_wrappers=[], 
                         render_mode="human", 
                         novice=novice,
                         debug=True)
    
    try:
        # Display environment info
        print("\n=== VISUALIZATION MODE ===")
        print("Controls:")
        print("  P: Pause/resume")
        print("  S: Toggle step mode")
        print("  SPACE: Advance one step (in step mode)")
        print("  Q: Quit")
        print("\nEnvironment rewards config:", env.rewards_dict)
        
        # Main visualization loop
        for round_idx in range(1, num_rounds + 1):
            # Start new episode
            env.reset()
            manager.reset()
            
            terminated_episode = False
            
            # Track action and observation for each agent
            prev_action = None
            prev_observation = None
            
            # Store last seen observations for each agent
            agent_observations = {}
            
            print(f"\n=== Starting Episode {round_idx} ===")
            
            # Per-agent loop
            for agent in env.agent_iter():
                # Handle visualization events (keyboard controls, etc.)
                if not viz.handle_events(env):
                    return
                
                # Handle pause state if paused
                if not viz.handle_pause(env):
                    return
                
                # Get current observation, reward, etc.
                observation, reward, termination, truncation, info = env.last()
                
                # Store observation for this agent
                if observation is not None:
                    agent_observations[agent] = observation.copy()
                
                # Check for capture condition
                if termination and not terminated_episode:
                    # Record termination rewards
                    terminated_episode = reward_tracker.record_termination_rewards(
                        env, agent_observations, manager)
                
                # Record regular rewards during normal gameplay
                if prev_action is not None and prev_observation is not None and not terminated_episode:
                    reward_tracker.record_step_reward(agent, reward, prev_observation, manager)
                
                # Handle death/truncation
                if termination or truncation:
                    action = None
                    prev_action = None
                else:
                    # Select a new action
                    action = manager.rl(observation)
                    # Store for next iteration
                    prev_action = action
                    prev_observation = observation
                
                # Step the environment
                env.step(action)
                
                # Auto-pause if needed
                viz.auto_pause_if_needed()
            
            # Final update at episode end
            manager.update()
            
            # Process end of episode
            reward_tracker.end_episode(round_idx, num_rounds, manager)
            
    finally:
        # Cleanup
        reward_tracker.close()
        viz.shutdown(env)
    
    # Final summary
    reward_tracker.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GridWorld environment with RL agent")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to run")
    parser.add_argument("--quiet", action="store_true", help="Disable detailed reward output")
    
    args = parser.parse_args()
    
    visualize(num_rounds=args.episodes, detailed_rewards=not args.quiet) 