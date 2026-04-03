import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size=1024,
        llm_hidden_size=1536,
        intermediate_size=4096
    ):
        super().__init__()

        self.norm = nn.LayerNorm(vision_hidden_size)

        self.gate_proj = nn.Linear(vision_hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(vision_hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, llm_hidden_size, bias=False)

        self.act_fn = F.silu

    def forward(self, x):
        x = self.norm(x)

        gate = self.act_fn(self.gate_proj(x))
        up   = self.up_proj(x)

        x = gate * up
        x = self.down_proj(x)

        return x