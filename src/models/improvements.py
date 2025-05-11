import math
import torch
from torch import nn
from torch.nn import functional as F

# 1. PrefixSumCounts
# Convert one‑hot tokens → running count of that same token position‑wise.
class PrefixSumCounts(nn.Module):
    def __init__(self, d_vocab):
        super().__init__()
        self.d_vocab = d_vocab

    def forward(self, x):
        x_long = x.long()
        one_hot = F.one_hot(x_long, num_classes=self.d_vocab).float()
        cumsum = torch.cumsum(one_hot, dim=1)
        idx = x_long.unsqueeze(-1)
        counts = torch.gather(cumsum, 2, idx).float()
        return counts

# 2. SparseExpertCountingNetwork
# Sparse mixture of experts for counting-related tasks
# MoE layer learns to pick one of several purpose‑built counting experts per token
class HistogramExpert(nn.Module):
    def forward(self, x):
        return x.sum(-1, keepdim=True)

class FrequencyExpert(nn.Module):
    def forward(self, x):
        total = x.sum(-1, keepdim=True) + 1e-6
        maxcount,_ = x.max(-1, keepdim=True)
        return maxcount / total

class UniquenessExpert(nn.Module):
    def forward(self, x):
        return (x != 0).float().sum(-1, keepdim=True)

class PatternCountExpert(nn.Module):
    def forward(self, x):
        diff = (x[:, 1:] != x[:, :-1]).float()
        return diff.sum(-1, keepdim=True)
    
class IdentityExpert(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 1, device=x.device)

class SparseExpertCountingNetwork(nn.Module):
    def __init__(self, hist_dim):
        super().__init__()
        self.experts = nn.ModuleList([
            HistogramExpert(),
            FrequencyExpert(),
            UniquenessExpert(),
            PatternCountExpert(),
            IdentityExpert(),
        ])
        self.router = nn.Linear(hist_dim, len(self.experts))
        with torch.no_grad():
            nn.init.constant_(self.router.bias, 0.0)
            identity_idx = len(self.experts) - 1
            self.router.bias[identity_idx] = 1.0

    def forward(self, histograms):
        logits = self.router(histograms)
        probs = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)
        if not self.training:
            argmax_idx = probs.argmax(dim=1, keepdim=True)
            hard = torch.zeros_like(probs).scatter_(1, argmax_idx, 1.0)
            routes = hard.detach() + probs - probs.detach()
        else:
            routes = probs
        expert_outs = torch.stack([exp(histograms) for exp in self.experts], dim=1)
        out = (routes.unsqueeze(-1) * expert_outs).sum(dim=1)
        return F.relu(out).squeeze(-1)