"""Graph Convolutional Network with Gated Recurrent Unit (GCNGRU) model.

This module defines the GCNGRU architecture, which combines Graph Convolutional Networks
(GCN) for spatial feature learning with Gated Recurrent Units (GRU) for temporal
sequence modeling. The model is designed for spatio-temporal forecasting tasks.

The architecture processes input sequences by:
1. Applying GCN layers to each time step independently for spatial feature extraction
2. Processing the temporal sequence through GRU layers
3. Generating multi-horizon predictions using a linear output layer

Classes
-------
GCNGRU
    Neural network model combining GCN and GRU for spatio-temporal prediction.

Notes
-----
This model is particularly suited for weather forecasting and other applications
requiring both spatial and temporal modeling on graph-structured data.
"""
###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# purpose: to define model class (GCN + GRU combo)
###########################################################################################

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNGRU(nn.Module):
    """Graph Convolutional Network with Gated Recurrent Unit model.

    This model combines Graph Convolutional Networks (GCN) for spatial feature
    learning with Gated Recurrent Units (GRU) for temporal sequence modeling.
    It is designed for spatio-temporal forecasting tasks on graph-structured data.

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_dim : int
        Dimension of hidden representations in GCN and GRU layers.
    num_nodes : int
        Number of nodes in the graph.
    out_channels : int
        Number of output features per node (e.g., 1 for temperature).
    num_layers : int, optional
        Number of GRU layers, by default 1.
    pred_horizons : int, optional
        Number of prediction time steps into the future, by default 5.

    Attributes
    ----------
    gcn : GCNConv
        Graph convolutional layer for spatial feature extraction.
    gru : nn.GRU
        Gated recurrent unit for temporal modeling.
    linear : nn.Linear
        Linear output layer for multi-horizon predictions.
    num_nodes : int
        Number of nodes in the graph.
    out_channels : int
        Number of output features per node.
    pred_horizons : int
        Number of prediction time steps.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_nodes: int,
        out_channels: int,
        num_layers: int = 1,
        pred_horizons: int = 5,
    ) -> None:
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, pred_horizons * out_channels)
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        self.pred_horizons = pred_horizons

    def forward(
        self, x_seq: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the GCNGRU model.

        Parameters
        ----------
        x_seq : torch.Tensor
            Input sequence tensor of shape (batch_size, timesteps, num_nodes, features).
        edge_index : torch.Tensor
            Graph connectivity in COO format, shape (2, num_edges).
        edge_weight : torch.Tensor
            Edge weights for the graph, shape (num_edges,).

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, pred_horizons, num_nodes, out_channels).
        """
        b, t, n, f = x_seq.shape
        assert self.num_nodes == n

        gcn_out_list: list[torch.Tensor] = []
        for time_step in range(t):
            x_t = x_seq[:, time_step, :, :].reshape(-1, f)  # [B*N, F] (F = input feats)
            out = self.gcn(x_t, edge_index)  # [B*N, H]
            out = out.reshape(b, n, -1)  # [B, N, H]
            gcn_out_list.append(out)

        gcn_out = torch.stack(
            gcn_out_list, dim=1
        )  # [B, T, N, H] (T = input window length)
        gcn_out = gcn_out.permute(0, 2, 1, 3)  # [B, N, T, H]
        gcn_out = gcn_out.reshape(b * n, t, -1)

        gru_out, _ = self.gru(gcn_out)  # [B*N, T, H]
        last = gru_out[:, -1, :]  # [B*N, H]
        out = self.linear(last)  # [B*N, pred_horizons * out_channels]
        return out.view(b, n, self.pred_horizons, self.out_channels).permute(
            0, 2, 1, 3
        )  # [B, H, N, D] -> [batch size, # horizons, # nodes, output feats (1, temp)]
