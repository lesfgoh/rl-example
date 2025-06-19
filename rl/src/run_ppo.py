# rl/src/run_ppo.py

from ppo_manager import PPOManager

def main():
    # adjust these hyperparameters if you like
    lr = 1e-4
    gamma = 0.99
    total_timesteps = 1_000_000

    manager = PPOManager(lr=lr, gamma=gamma)
    manager.train(total_timesteps=total_timesteps)

if __name__ == "__main__":
    main()