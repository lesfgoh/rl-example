import torch
from ..checkpoint.checkpoint import load_checkpoint, save_checkpoint
from ..cnn.model import PolicyNet

def reset_guard_network(checkpoint_dir: str = "checkpoints", filename: str = "latest.pt"):
    # Load the existing checkpoint
    data = load_checkpoint(checkpoint_dir, filename)
    
    # Create a new guard network with random initialization
    guard_net = PolicyNet(n_actions=4, history_len=4)
    guard_optim = torch.optim.Adam(guard_net.parameters(), lr=1e-4)
    
    # Create a minimal scout network (we won't use it, just need valid parameters)
    scout_net = PolicyNet(n_actions=4, history_len=4)
    scout_optim = torch.optim.Adam(scout_net.parameters(), lr=1e-4)
    
    # Save the checkpoint with new guard network but same scout network
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        scout_net=scout_net,  # Use minimal network instead of empty module
        scout_optim=scout_optim,  # Use valid optimizer
        guard_net=guard_net,
        guard_optim=guard_optim,
        filename=filename,
        scout_state_dict=data["scout_state_dict"],  # This will override the network's parameters
        scout_optim_state=data["scout_optim_state"],  # This will override the optimizer's state
    )
    print(f"Successfully reset guard network while preserving scout network in {checkpoint_dir}/{filename}")

if __name__ == "__main__":
    # When running as a script, we need to run it as a module
    import sys
    from pathlib import Path
    # Add the src directory to Python path
    src_dir = str(Path(__file__).parent.parent.parent)
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    reset_guard_network() 