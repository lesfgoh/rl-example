import numpy as np
import struct
from environment.gridworld import raw_env

def unpack_bits(viewcone):
    """
    Unpacks the viewcone bits into a 3D numpy array.
    """

    viewcone_shape = np.shape(viewcone)

    unpacked = np.zeros((viewcone_shape[0], viewcone_shape[1], 7), dtype=object)
    for row_idx, row in enumerate(viewcone):
        for tile_idx, tile in enumerate(row):

            tile_contains = tile & 0b11

            if tile_contains == 0:
                tile_contains = "no vision"
            elif tile_contains == 1:
                tile_contains = "empty"
            elif tile_contains == 2:
                tile_contains = "recon"
            elif tile_contains == 3:
                tile_contains = "mission"
            else:
                print("Error: tile contains unknown value")

            unpacked[row_idx][tile_idx][0] = tile_contains

            for i in range(2, 8):
                tile_type = (tile >> i) & 1
                unpacked[row_idx][tile_idx][i-1] = tile_type==1

    return unpacked

# 1) Create and reset
env = raw_env(render_mode="human", debug=False, novice=False)
env.reset(seed=100)

# 2) Pick one agent (e.g. the scout goes first)
agent = env.scout

# 3b) Render initial frame and wait for user
# env.render()
# input("Press Enter to continue after viewing the initial map...")

# 4) Extract that agent’s observation
obs = env.observe(agent)
viewcone = obs["viewcone"]

print(unpack_bits(viewcone))

print(f"\nAgent {agent} at {obs['location']}, facing {obs['direction']}")
print(obs)

# 4) Take the move forward action
action=0
env.step(action)
env.step(4) # This is B
env.step(4) # This is A
env.step(4)


# Render and wait to show the result of the move-forward
env.render()
input("Press Enter to continue after move-forward...")

# 5) Now pull out the new data by hand:
new_obs    = env.observe(agent)
# Retrieve the last reward for this agent from the env's last_rewards dict
reward = env.rewards[agent]                  
info       = {}                              

# 6) Render + print

print("New obs after moving forward:")
print(new_obs)
print(f"Step count now: {new_obs['step']}")
print(f"Reward={reward}")

# move backward
action=1
env.step(action)
env.step(4)
env.step(4)
env.step(4)


# Render and wait to show the result of the move-backward
env.render()
input("Press Enter to close the window...")

# 5) Now pull out the new data by hand:
new_obs    = env.observe(agent)
# Retrieve the last reward for this agent from the env's last_rewards dict
reward = env.rewards[agent]                  
info       = {}                              

print("New obs after moving backwards:")
print(new_obs)
print(f"Step count now: {new_obs['step']}")
print(f"Reward={reward}")
