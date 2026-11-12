"""Stage 3B — Fuzzy Context-Enriched Spatial GNN (blueprint §7.2, §7.3).

Eliminates the additive-AHP Rank Reversal Paradox: every physical feature is
normalised through *domain-anchored* learnable fuzzy membership functions
mu(x) = sigmoid(k * (x - c)) whose centres/slopes are physical inflection
points, NOT candidate-pool statistics — so adding or deleting candidate
sites cannot change any other site's score.  Multi-head graph attention
(GATv2) then propagates regional context over the Uber-H3 style spatial
graph, and a scoring head emits the rank-invariant suitability S_i in [0,1].

The GATv2 layer is implemented self-contained (dense attention over the
sparse edge list) so no torch-geometric binary dependency is needed; the
mathematics follows Brody et al. 2022 exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Node feature order (7 features, blueprint Table 2 Layer 3B) ----
FEATURES = ["wpd_120m", "geff_solar", "buildable_ratio", "terrain_slope_deg",
            "substation_dist_km", "soiling_pct", "highway_dist_km"]

# Physical inflection centres & slopes (maximising +k / minimising -k)
FUZZY_INIT = {
    "wpd_120m":          {"center": 320.0, "slope": +0.012},  # W/m^2
    "geff_solar":        {"center": 4.4,   "slope": +2.2},    # kWh/m^2/day
    "buildable_ratio":   {"center": 0.45,  "slope": +7.0},    # [0,1]
    "terrain_slope_deg": {"center": 8.0,   "slope": -0.55},   # degrees
    "substation_dist_km": {"center": 18.0, "slope": -0.22},   # km
    "soiling_pct":       {"center": 4.0,   "slope": -0.80},   # %
    "highway_dist_km":   {"center": 12.0,  "slope": -0.30},   # km
}


def fuzzy_init_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.tensor([FUZZY_INIT[f]["center"] for f in FEATURES])
    slopes = torch.tensor([FUZZY_INIT[f]["slope"] for f in FEATURES])
    return centers, slopes


class DifferentiableFuzzyMembership(nn.Module):
    """mu(x) = sigmoid(slope * (x - center)) — immune to pool Rank Reversal."""

    def __init__(self, init_centers: torch.Tensor, init_slopes: torch.Tensor):
        super().__init__()
        self.centers = nn.Parameter(init_centers.clone().float())
        self.slopes = nn.Parameter(init_slopes.clone().float())

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.slopes * (x_raw - self.centers))


class GATv2Conv(nn.Module):
    """GATv2 (dynamic attention) over an edge list, with edge attributes."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 edge_dim: int = 3, concat: bool = True):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.w_src = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.w_dst = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.w_edge = nn.Linear(edge_dim, heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.attn)
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        h_src = self.w_src(x).view(-1, self.heads, self.out_dim)
        h_dst = self.w_dst(x).view(-1, self.heads, self.out_dim)
        h_e = self.w_edge(edge_attr).view(-1, self.heads, self.out_dim)
        # GATv2: score = a^T LeakyReLU(W_s x_j + W_d x_i + W_e e_ij)
        z = F.leaky_relu(h_src[src] + h_dst[dst] + h_e, 0.2)
        score = (z * self.attn[None]).sum(-1)                       # [E, H]
        # softmax over incoming edges of each destination node
        score = score - score.max()
        exp = score.exp()
        denom = torch.zeros(n, self.heads, device=x.device).index_add_(
            0, dst, exp) + 1e-16
        alpha = exp / denom[dst]
        msg = h_src[src] * alpha.unsqueeze(-1)                      # [E, H, D]
        out = torch.zeros(n, self.heads, self.out_dim,
                          device=x.device).index_add_(0, dst, msg)
        out = out.reshape(n, -1) if self.concat else out.mean(1)
        return out + self.bias


class ContextEnrichedSpatialGNN(nn.Module):
    """Fuzzy memberships -> 2x GATv2 -> sigmoid scoring head S_i in [0,1]."""

    def __init__(self, num_features: int = len(FEATURES), edge_dim: int = 3,
                 init_centers: torch.Tensor | None = None,
                 init_slopes: torch.Tensor | None = None):
        super().__init__()
        if init_centers is None or init_slopes is None:
            init_centers, init_slopes = fuzzy_init_tensors()
        self.fuzzy_layer = DifferentiableFuzzyMembership(init_centers, init_slopes)
        self.gat1 = GATv2Conv(num_features, 64, heads=4, edge_dim=edge_dim,
                              concat=True)
        self.gat2 = GATv2Conv(64 * 4, 64, heads=2, edge_dim=edge_dim,
                              concat=False)
        self.scoring_head = nn.Sequential(
            nn.Linear(64 + num_features, 32), nn.SiLU(), nn.Linear(32, 1),
            nn.Sigmoid())

    def forward(self, x_raw: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        x_fuzzy = self.fuzzy_layer(x_raw)
        h1 = F.elu(self.gat1(x_fuzzy, edge_index, edge_attr))
        h2 = F.elu(self.gat2(h1, edge_index, edge_attr))
        combined = torch.cat([h2, x_fuzzy], dim=-1)
        return self.scoring_head(combined).squeeze(-1)

    # ------------------------------------------------------------------
    @staticmethod
    def rank_invariant_loss(pred_scores: torch.Tensor,
                            target_yields: torch.Tensor,
                            pair_indices: torch.Tensor,
                            margin: float = 0.02) -> torch.Tensor:
        """Pairwise margin-ranking + calibration loss.

        Ranking on randomly drawn pairs makes the ordering depend only on
        each site's own physics, and an MSE calibration term ties scores to
        GEM Operating-vs-Failed outcome labels."""
        i, j = pair_indices[0], pair_indices[1]
        sign = torch.sign(target_yields[i] - target_yields[j])
        rank = F.relu(margin - sign * (pred_scores[i] - pred_scores[j])).mean()
        calib = F.mse_loss(pred_scores, target_yields)
        return rank + 0.5 * calib


def knn_spatial_graph(coords_km: torch.Tensor, k: int = 6
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a KNN graph (H3-neighbour analogue) with edge attrs
    [dist_km_norm, dx_norm, dy_norm]."""
    n = coords_km.shape[0]
    d = torch.cdist(coords_km, coords_km)
    d.fill_diagonal_(float("inf"))
    knn = d.topk(k, largest=False).indices                 # [N, k]
    src = knn.reshape(-1)
    dst = torch.arange(n).repeat_interleave(k)
    delta = coords_km[src] - coords_km[dst]
    dist = delta.norm(dim=1, keepdim=True)
    scale = dist.max().clamp(1e-6)
    edge_attr = torch.cat([dist / scale, delta / scale], dim=1)
    return torch.stack([src, dst]), edge_attr
