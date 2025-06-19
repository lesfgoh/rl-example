import numpy as np
import torch
import torch.nn.functional as F
from pprint import pprint

MAX_MAP_X = 9 
MAX_MAP_Y = 12
NUM_DIRECTIONS = 4 # 0: right, 1: down, 2: left, 3: up
MAX_STEPS = 100

def preprocess_observation(observation_dict):
    """
    Converts raw observation into tensors:
      - 'viewcone': FloatTensor of shape (8,7,5), one bit-plane per channel.
      - 'vector_features': FloatTensor of shape (8,), concatenation of
         [one-hot direction (4), normalized location (2), scout flag (1), normalized step (1)].
    """
    # Unpack viewcone bits into 8 channels via numpy.unpackbits
    raw = np.array(observation_dict["viewcone"], dtype=np.uint8) 
    bits = np.unpackbits(raw[:, :, None], axis=2)     
    viewcone = torch.from_numpy(bits.transpose(2, 0, 1)).float()  

    # Build vector features
    #   direction one-hot
    dir_idx = torch.tensor(observation_dict["direction"], dtype=torch.int64)
    direction = F.one_hot(dir_idx, NUM_DIRECTIONS).float()        

    #   normalized location
    loc = torch.tensor(observation_dict["location"], dtype=torch.float32)
    max_xy = torch.tensor([MAX_MAP_X, MAX_MAP_Y], dtype=torch.float32)
    loc = (loc / max_xy).clamp(0.0, 1.0)                          

    #   scout flag
    scout = torch.tensor([observation_dict["scout"]], dtype=torch.float32)  

    #   normalized step
    step = torch.tensor([observation_dict["step"] / MAX_STEPS], dtype=torch.float32).clamp(0.0, 1.0)  

    vector_features = torch.cat([direction, loc, scout, step], dim=0)

    return {"viewcone": viewcone, "vector_features": vector_features}

if __name__ == '__main__':
    # Test:
    sample_obs = {
        'viewcone': [[  0,   0,   0,   0,   0],
                     [  0,   0,   0,   0,   0],
                     [  0,   0, 197,  66,  98],
                     [  0,   0, 129,   2,   2],
                     [  0,   0, 130,   3,  51],
                     [  0,   0, 130,  34,   0],
                     [  0,   0, 146,   2,   0]], 
       'direction': np.int64(0), 
       'location': [0, 0], 
       'scout': 1, 'step': 2}
    
    processed = preprocess_observation(sample_obs)
    pprint(processed)
    print("Processed Viewcone Shape:", processed["viewcone"].shape)
    print("Processed Viewcone:\n", processed["viewcone"])
    print("Processed Vector Features:", processed["vector_features"])
    print("Processed Vector Features Shape:", processed["vector_features"].shape)