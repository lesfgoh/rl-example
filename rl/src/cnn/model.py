import torch.nn as nn
import torch.nn.functional as F
import torch

class PolicyNet(nn.Module):
    def __init__(self, n_actions: int, history_len: int = 1):
        super().__init__()
        in_ch  = 8 * history_len
        vf_dim = 8 * history_len

        # --- Convolutional tower ---
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),  # keeps 7×5
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),     # -> 5×3
            nn.ReLU(),
            nn.Flatten(),
        )

        # --- MLP for vector features ---
        self.fc_vec = nn.Sequential(
            nn.Linear(vf_dim, 64),
            nn.ReLU(),
        )

        # --------- Build head dynamically ----------
        with torch.no_grad():
            dummy_vc = torch.zeros(1, in_ch, 7, 5)
            conv_out_dim = self.conv(dummy_vc).shape[1]   # 64 * 5 * 3 = 960

        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        # optional orthogonal init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))  # type: ignore
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, vc, vf):
        """
        vc: (B, 8*history_len, 7, 5)
        vf: (B, 8*history_len)
        """
        x1 = self.conv(vc)            # (B, conv_out_dim)
        x2 = self.fc_vec(vf)          # (B, 64)
        x  = torch.cat([x1, x2], dim=1)
        logits = self.head(x)         # (B, n_actions)
        return logits