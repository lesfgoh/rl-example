from typing import Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # This is to avoid circular import if your env type is complex
    # For PettingZoo, the base AECEnv might be sufficient or a custom protocol
    from pettingzoo.utils.env import AECEnv 

def get_agent_info(
    raw_agent_id: str,
    env: 'AECEnv', # Use type hint for the environment
    raw_observation: Dict[str, Any] | None = None
) -> Tuple[str, int, str]:
    """Processes a raw agent ID to a mapped ID and scout status.

    Args:
        raw_agent_id: The agent ID from the environment (e.g., "player_0").
        env: The environment instance, used to access env.scout.
        raw_observation: Optional raw observation dictionary for this agent.

    Returns:
        A tuple: (mapped_agent_id, is_scout_flag (0 or 1), player_index_str)
    """
    is_scout_player: bool
    
    # Primary source: observation dict
    if raw_observation is not None and 'scout' in raw_observation:
        is_scout_player = bool(raw_observation.get("scout"))
    # Secondary source: env.scout attribute
    elif hasattr(env, 'scout') and getattr(env, 'scout', None) is not None:
        is_scout_player = (raw_agent_id == getattr(env, 'scout'))
    # Fallback: guess from ID (least reliable)
    else:
        is_scout_player = "scout" in raw_agent_id.lower()
        print(f"Warning (get_agent_info): Could not reliably determine scout status for {raw_agent_id} from obs or env.scout. Guessed: {is_scout_player}")

    player_index_str = "0" # Default index
    if "_" in raw_agent_id:
        parts = raw_agent_id.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            player_index_str = parts[-1]
        # If raw_agent_id is like "scout" or "guard" without index, player_index_str remains "0"
        # Or if it's player_X, it gets X. If it's just "scout" and is_scout_player is true, index is "0".
    
    mapped_id: str
    if is_scout_player:
        mapped_id = f"scout_{player_index_str}"
    else:
        mapped_id = f"guard_{player_index_str}"
    
    scout_flag_int = 1 if is_scout_player else 0

    return mapped_id, scout_flag_int, player_index_str 