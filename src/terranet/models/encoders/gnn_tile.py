"""GNN encoder over the per-tile building graph.

Nodes = buildings (features: height, base area, volume, centroid offsets);
edges = k-NN over centroids with inverse-distance weights. Mean-pooled to a
tile embedding. Captures street-canyon structure the raster can blur.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, global_mean_pool


class TileGNN(nn.Module):
    def __init__(self, node_dim: int = 5, hidden: int = 128, emb_dim: int = 256, layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList(
            [GraphConv(node_dim if i == 0 else hidden, hidden) for i in range(layers)]
        )
        self.out = nn.Linear(hidden, emb_dim)
        self.act = nn.GELU()

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        return self.out(global_mean_pool(x, batch))
