###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# purpose: to define model class (GCN + GRU combo)
###########################################################################################

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNGRU(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_nodes, out_channels, num_layers=1, pred_horizons=5):
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, pred_horizons * out_channels)
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        self.pred_horizons = pred_horizons

    def forward(self, x_seq, edge_index, edge_weight):
        B, T, N, F = x_seq.shape
        assert N == self.num_nodes

        gcn_out = []
        for t in range(T):
            x_t = x_seq[:, t, :, :].reshape(-1, F)  # [B*N, F] (F = input feats)
            out = self.gcn(x_t, edge_index)         # [B*N, H]
            out = out.reshape(B, N, -1)             # [B, N, H]
            gcn_out.append(out)

        gcn_out = torch.stack(gcn_out, dim=1)       # [B, T, N, H] (T = input window length)
        gcn_out = gcn_out.permute(0, 2, 1, 3)       # [B, N, T, H]
        gcn_out = gcn_out.reshape(B * N, T, -1)

        gru_out, _ = self.gru(gcn_out)              # [B*N, T, H]
        last = gru_out[:, -1, :]                    # [B*N, H]
        out = self.linear(last)                     # [B*N, pred_horizons * out_channels]
        out = out.view(B, N, self.pred_horizons, self.out_channels).permute(0, 2, 1, 3)
        return out  # [B, H, N, D] -> [batch size, # horizons, # nodes, output feats (1, temp)]
