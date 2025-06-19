import os
import torch
from typing import Any, Dict, Optional

DEFAULT_FILENAME = ("latest.pt")


def save_checkpoint(
    policy_dir: str,
    scout_net: torch.nn.Module,
    scout_optim: torch.optim.Optimizer,
    guard_net: torch.nn.Module,
    guard_optim: torch.optim.Optimizer,
    filename: str = DEFAULT_FILENAME,
    scout_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    scout_optim_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save scout and guard network+optimizer states to a single file.
    Optionally accepts scout state dictionaries directly to preserve them.
    """
    os.makedirs(policy_dir, exist_ok=True)
    path = os.path.join(policy_dir, filename)
    data = {
        "scout_state_dict": scout_state_dict
        if scout_state_dict is not None
        else scout_net.state_dict(),
        "scout_optim_state": scout_optim_state
        if scout_optim_state is not None
        else scout_optim.state_dict(),
        "guard_state_dict": guard_net.state_dict(),
        "guard_optim_state": guard_optim.state_dict(),
    }
    torch.save(data, path)
    print(f"[checkpoint] Saved checkpoint to {path}")


def load_checkpoint(
    checkpoint_dir: str,
    filename: str = DEFAULT_FILENAME,
    map_location: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint dict from file.
    Raises FileNotFoundError if the file does not exist.
    """
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    data = torch.load(path, map_location=map_location)
    print(f"[checkpoint] Loaded checkpoint from {path}")
    return data
