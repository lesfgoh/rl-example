from collections import deque
import torch

class FrameStack:
    """
    Maintains a fixed‐length history of recent observations (both
    “viewcone” and “vector_features”) and produces a single stacked
    tensor for each when asked.
    """

    def __init__(self, history_len: int, device: torch.device):
        """
        :param history_len: how many past frames to keep (>=1).
        :param device:      torch.device to which tensors should be moved.
        """
        if history_len < 1:
            raise ValueError("history_len must be >= 1")
        self.history_len = history_len
        self.device = device

        # deques of raw (unbatched) tensors
        self._vc_deque = deque(maxlen=history_len)
        self._vf_deque = deque(maxlen=history_len)

    def reset(self):
        """
        Clear any previous history and seed with the initial observation.
        :param initial_vc: viewcone tensor of shape (C,H,W)
        :param initial_vf: vector_features tensor of shape (D,)
        """
        self._vc_deque.clear()
        self._vf_deque.clear()

    def append(self, vc: torch.Tensor, vf: torch.Tensor):
        """
        Add one new frame to history.
        :param vc: viewcone tensor (C,H,W)
        :param vf: vector_features tensor (D,)
        """
        # ensure correct device
        vc = vc.to(self.device)
        vf = vf.to(self.device)
        self._vc_deque.append(vc)
        self._vf_deque.append(vf)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two batched tensors, each of shape (1, C*history_len, H, W)
        and (1, D*history_len) respectively, by stacking the last
        history_len frames. If not enough frames have been seen yet,
        the first frame is repeated.
        """
        # pad if necessary
        if len(self._vc_deque) < self.history_len:
            first_vc = self._vc_deque[0]
            first_vf = self._vf_deque[0]
            while len(self._vc_deque) < self.history_len:
                self._vc_deque.appendleft(first_vc)
                self._vf_deque.appendleft(first_vf)

        # stack along channel/feature dims
        vc_stack = torch.cat(list(self._vc_deque), dim=0).unsqueeze(0)
        vf_stack = torch.cat(list(self._vf_deque), dim=0).unsqueeze(0)
        return vc_stack, vf_stack

    def __len__(self) -> int:
        """Number of frames currently stored (≤ history_len)."""
        return len(self._vc_deque)