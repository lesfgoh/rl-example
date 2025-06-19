#!/usr/bin/env python3
"""
Main entry point for RL training, supporting both visualization and headless modes.
"""

import os
import sys
import argparse
from visualize import visualize
from train_headless import train_headless


def main():
    """
    Parse command-line arguments and launch the appropriate training mode.
    """
    parser = argparse.ArgumentParser(
        description="Train RL agent for GridWorld environment"
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Run in visualization mode with interactive controls",
    )
    parser.add_argument(
        "--debug-human",
        action="store_true",
        help="Uses human render mode for headless training",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed reward output",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=100,
        help="How often to report progress (in episodes)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000000000,
        help="Number of episodes to run",
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
    parser.add_argument(
        "--history-len",
        type=int,
        default=4,
        help="How long to retain previous viewcones",
    )

    args = parser.parse_args()

    if args.visual:
        # Run in visualization mode
        print("Running in visualization mode with interactive controls")
        visualize(num_rounds=args.episodes, detailed_rewards=args.verbose)
    else:
        # Run in headless mode for faster training
        print("Running in headless mode for faster training")
        train_headless(
            num_rounds=args.episodes,
            detailed_rewards=args.verbose,
            checkpoint_interval=args.report_interval,
            entropy_beta=args.entropy_beta,
            checkpoint_save_interval=args.checkpoint_save_interval,
            histogram_log_interval=args.histogram_log_interval,
            history_len=args.history_len,
            debug_human=args.debug_human,
        )


if __name__ == "__main__":
    main()
