#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

"""
    model.py
    This file contains the model used for the Graph Autoencoder (GAE) model trained using unsupervised learning.
    The model is composed by a GNN-based encoder and a NN-based decoder, with the first using node feature vectors, topology features, and edge features to encode the graph structure,
     and the second reconstructing several elements in order to ensure the graph structure is preserved.
"""

import torch
import os
from torch_geometric.nn import GCNConv, GATConv, EdgeConv, NNConv, SAGEConv, GINConv
from torch.nn import ReLU, ModuleList, Sequential, Linear
import sys
from torch_geometric.nn.norm import BatchNorm
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
torch.set_default_dtype(torch.float32)

# Encoder GNN-based architecture used to encode the graph structure
class GAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, cfg_layers, edge_feature_dim):
        super(GAEEncoder, self).__init__()
        self.layers = ModuleList()
        self.in_channels = in_channels
        # Custom number of layers of several types
        for layer_cfg in cfg_layers:
            if layer_cfg['type'] == 'GCNConv':
                self.layers.append(GCNConv(in_channels, layer_cfg['out_channels']))
                in_channels = layer_cfg['out_channels']
            elif layer_cfg['type'] == 'GATConv':
                self.layers.append(GATConv(in_channels, layer_cfg['out_channels'], heads=layer_cfg.get('heads', 1),
                                           concat=layer_cfg.get('concat', True)))
                in_channels = layer_cfg['out_channels'] * layer_cfg.get('heads', 1) if layer_cfg.get('concat',
                                                                                                     True) else \
                layer_cfg['out_channels']
            elif layer_cfg['type'] == 'GINConv':
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, layer_cfg['out_channels']),
                    torch.nn.ReLU(),
                    torch.nn.Linear(layer_cfg['out_channels'], layer_cfg['out_channels'])
                )
                self.layers.append(GINConv(mlp, train_eps=layer_cfg.get('train_eps', False)))
                in_channels = layer_cfg['out_channels']
            elif layer_cfg['type'] == 'EdgeConv':
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(2 * in_channels, layer_cfg['out_channels']),
                    torch.nn.ReLU(),
                    torch.nn.Linear(layer_cfg['out_channels'], layer_cfg['out_channels'])
                )
                self.layers.append(EdgeConv(mlp, aggr=layer_cfg.get('aggr', 'mean')))
                in_channels = layer_cfg['out_channels']
            elif layer_cfg['type'] == 'SAGEConv':
                self.layers.append(
                    SAGEConv(
                        in_channels,
                        layer_cfg['out_channels'],
                        aggr=layer_cfg.get('aggr', 'mean'),
                        normalize=layer_cfg.get('normalize', False),
                        project=layer_cfg.get('project', False),
                        bias=layer_cfg.get('bias', True)
                    )
                )
                in_channels = layer_cfg['out_channels']
            elif layer_cfg['type'] == 'NNConv':
                # this layer ensures also usage of edge features
                edge_network = Sequential(
                    Linear(edge_feature_dim, layer_cfg['NN_channels']),
                    ReLU(),
                    Linear(layer_cfg['NN_channels'], in_channels * layer_cfg['out_channels'])
                )

                self.layers.append(NNConv(in_channels, layer_cfg['out_channels'], edge_network))
                in_channels = layer_cfg['out_channels']
            self.layers.append(BatchNorm(in_channels))
            activation = layer_cfg.get('activation', 'ReLU')
            if activation == 'ReLU':
                self.layers.append(ReLU())
            elif activation == 'Sigmoid':
                self.layers.append(torch.nn.Sigmoid())
            elif activation == 'Tanh':
                self.layers.append(torch.nn.Tanh())
            elif activation == 'LeakyReLU':
                self.layers.append(torch.nn.LeakyReLU())
            # otherwise not adding any activation
        self.embedding_dim = in_channels  # Last layer output channels

    def forward(self, x, edge_index, edge_attr=None):
        # Forwarding based on the type of layer
        for layer in self.layers:
            if isinstance(layer, (GCNConv, GATConv, SAGEConv, GINConv)):
                x = layer(x, edge_index)
            elif isinstance(layer, NNConv):
                if edge_attr is not None:
                    x = layer(x, edge_index, edge_attr)
                else:
                    num_edges = edge_index.size(1)
                    edge_attr = torch.zeros((num_edges, 0), device=x.device)  # Default edge_attr
                    x = layer(x, edge_index, edge_attr)
            elif isinstance(layer, EdgeConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

# NN-based decoder used to reconstruct characteristics based on node embeddings
class GAEDecoder(nn.Module):
    def __init__(self, out_channels, binary_indices, multi_class_info, multi_class_info_order, continuous_indices, ranking_indices,
                 edge_binary_indices, edge_continuous_indices,  edge_multi_class_info, edge_multi_class_info_order, edge_ranking_indices):
        super(GAEDecoder, self).__init__()

        # Setup for binary, multi-class categorical data, and continuous data
        self.binary_indices = binary_indices
        self.multi_class_info = multi_class_info
        self.multi_class_info_order = multi_class_info_order
        self.continuous_indices = continuous_indices
        self.edge_binary_indices = edge_binary_indices
        self.edge_continuous_indices = edge_continuous_indices
        self.edge_multi_class_info = edge_multi_class_info
        self.edge_multi_class_info_order = edge_multi_class_info_order
        self.ranking_indices = ranking_indices  # list of feature indices for which we only care about order
        self.edge_ranking_indices = edge_ranking_indices

        # Decoder for node features
        if len(binary_indices) > 0:
            self.binary_feature_decoder = nn.Sequential(
                nn.Linear(out_channels, len(self.binary_indices)), # one output neuron per binary feature
                nn.Sigmoid() # sigmoid applied to ensure values are between 0 and 1
            )

        if sum(multi_class_info.values()) > 0:
            self.multi_class_feature_decoder = nn.Linear(out_channels, sum(multi_class_info.values())) # one output neuron per each class summing up to the total number of classes

        if len(continuous_indices) > 0:
            self.cont_feature_decoder = nn.Sequential(
                nn.Linear(out_channels, len(continuous_indices))
            )

        if len(self.ranking_indices) > 0:
            self.ranking_feature_decoder = nn.Linear(out_channels, len(ranking_indices))

        # Decoder for adjacency matrix
        self.adj_decoder = nn.Sequential(
            nn.Linear(2 * out_channels, 1),
            nn.Sigmoid()
        )

        # Decoder for edge features
        if len(edge_binary_indices) > 0:
            self.edge_binary_feature_decoder = nn.Sequential(
                nn.Linear(2 * out_channels, len(edge_binary_indices)), # takes the concatenation of the two node embeddings
                nn.Sigmoid() # sigmoid applied to ensure values are between 0 and 1
            )

        if len(edge_continuous_indices) > 0:
            self.edge_continuous_feat_decoder = nn.Sequential(
                nn.Linear(2 * out_channels, len(edge_continuous_indices)), # takes the concatenation of the two node embeddings
            )

        if sum(edge_multi_class_info.values()) > 0:
            self.edge_multi_class_feature_decoder = nn.Linear(2 * out_channels, sum(edge_multi_class_info.values()))


        if len(edge_ranking_indices) > 0:
            self.edge_ranking_feature_decoder = nn.Linear(2 * out_channels, len(edge_ranking_indices))


    def forward(self, z, edge_index):
        # Initialize output tensor for all features
        total_outputs = len(self.binary_indices) + sum(self.multi_class_info.values()) + len(self.continuous_indices)
        reconstructed_x = torch.zeros((z.size(0), total_outputs), device=z.device)

        # Decode node features
        if len(self.binary_indices) > 0:
            binary_features = self.binary_feature_decoder(z)
            reconstructed_x[:, :len(self.binary_indices)] = binary_features

        offset =  len(self.binary_indices)
        if sum(self.multi_class_info.values()) > 0:
            multi_class_logits = self.multi_class_feature_decoder(z)
            # Softmax applied segment-wise for each categorical feature
            start = 0
            for attribute in self.multi_class_info_order:
                num_classes = self.multi_class_info[attribute]
                end = start + num_classes
                reconstructed_x[:, offset + start:offset + end] = F.softmax(multi_class_logits[:, start:end], dim=1)
                start += num_classes

        if len(self.continuous_indices) > 0:
            continuous_features = self.cont_feature_decoder(z)
            reconstructed_x[:, -len(self.continuous_indices):] = continuous_features

        if len(self.ranking_indices) > 0:
            ranking_features = self.ranking_feature_decoder(z)
        else:
            ranking_features = None

        # Reconstruct adjacency matrix
        num_nodes = z.size(0)
        row_idx, col_idx = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
        edge_pairs = torch.cat([z[row_idx.reshape(-1)], z[col_idx.reshape(-1)]], dim=1)  # [N^2, 2*out_channels]

        # Binary adjacency
        adj_flat = self.adj_decoder(edge_pairs).view(num_nodes, num_nodes)
        adj = torch.sigmoid(adj_flat)  # probabilities for edge existence

        # Decode edge features
        edge_embeddings = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        edge_multi_class_len = 0
        for attribute in self.edge_multi_class_info_order:
            edge_multi_class_len += self.edge_multi_class_info[attribute]
        reconstructed_edge_features = torch.zeros(
            (edge_index.size(1), len(self.edge_binary_indices) + edge_multi_class_len + len(self.edge_continuous_indices)), device=z.device)

        if len(self.edge_binary_indices) > 0:
            edge_binary_features = self.edge_binary_feature_decoder(edge_embeddings)
            reconstructed_edge_features[:, :len(self.edge_binary_indices)] = edge_binary_features

        if len(self.edge_multi_class_info) > 0:
            edge_multi_class_logits = self.edge_multi_class_feature_decoder(edge_embeddings)
            # Softmax applied segment-wise for each categorical feature
            start = 0
            for attribute in self.edge_multi_class_info_order:
                num_classes = self.edge_multi_class_info[attribute]
                end = start + num_classes
                reconstructed_edge_features[:, len(self.edge_binary_indices) + start:len(self.edge_binary_indices) + end] = \
                    F.softmax(edge_multi_class_logits[:, start:end], dim=1)
                start += num_classes

        if len(self.edge_continuous_indices) > 0:
            # Continuous features are reconstructed from the concatenation of the two node embeddings
            edge_continuous_features = self.edge_continuous_feat_decoder(edge_embeddings)
            reconstructed_edge_features[:, len(self.edge_binary_indices)+sum(self.edge_multi_class_info.values()):] = edge_continuous_features

        if len(self.edge_ranking_indices) > 0:
            edge_ranking_features = self.edge_ranking_feature_decoder(edge_embeddings)
        else:
            edge_ranking_features = None

        return reconstructed_x, adj, reconstructed_edge_features, ranking_features, edge_ranking_features

# Overall GAE combining the encoder and decoder
class GAE(torch.nn.Module):
    def __init__(self, in_channels, cfg_layers, edge_feat_dim, binary_indices, multi_class_info, multi_class_info_order, continuous_indices, ranking_indices, edge_binary_indices, edge_continuous_indices, edge_multi_class_info, edge_multi_class_info_order, edge_ranking_indices):
        super(GAE, self).__init__()
        self.encoder = GAEEncoder(in_channels, cfg_layers, edge_feat_dim)
        last_layer_output_channels = cfg_layers[-1]['out_channels']
        self.decoder = GAEDecoder(last_layer_output_channels, binary_indices, multi_class_info, multi_class_info_order, continuous_indices, ranking_indices, edge_binary_indices, edge_continuous_indices, edge_multi_class_info, edge_multi_class_info_order, edge_ranking_indices)

    def forward(self, x, edge_index, edge_attr):
        z = self.encoder(x, edge_index, edge_attr)  # Node embeddings
        reconstructed_x, reconstructed_adj, reconstructed_edge_features, ranking_features, edge_ranking_features = self.decoder(z, edge_index)  # Reconstructed elements
        return reconstructed_x, reconstructed_adj, reconstructed_edge_features, ranking_features, edge_ranking_features

class VGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, cfg_layers, edge_feature_dim, latent_dim=None):
        super(VGAEEncoder, self).__init__()
        self.layers = ModuleList()
        self.in_channels = in_channels

        # same as your original encoder
        for layer_cfg in cfg_layers:
            if layer_cfg['type'] == 'GCNConv':
                self.layers.append(GCNConv(in_channels, layer_cfg['out_channels']))
                in_channels = layer_cfg['out_channels']
            elif layer_cfg['type'] == 'GATConv':
                self.layers.append(GATConv(in_channels, layer_cfg['out_channels'], heads=layer_cfg.get('heads', 1),
                                           concat=layer_cfg.get('concat', True)))
                in_channels = layer_cfg['out_channels'] * layer_cfg.get('heads', 1) if layer_cfg.get('concat',
                                                                                                     True) else \
                    layer_cfg['out_channels']
            elif layer_cfg['type'] == 'EdgeConv':
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(2 * in_channels, layer_cfg['out_channels']),
                    torch.nn.ReLU(),
                    torch.nn.Linear(layer_cfg['out_channels'], layer_cfg['out_channels'])
                )
                self.layers.append(EdgeConv(mlp, aggr=layer_cfg.get('aggr', 'mean')))
                in_channels = layer_cfg['out_channels']
            elif layer_cfg['type'] == 'NNConv':
                # this layer ensures also usage of edge features
                edge_network = Sequential(
                    Linear(edge_feature_dim, layer_cfg['NN_channels']),
                    ReLU(),
                    Linear(layer_cfg['NN_channels'], in_channels * layer_cfg['out_channels'])
                )

                self.layers.append(NNConv(in_channels, layer_cfg['out_channels'], edge_network))
                in_channels = layer_cfg['out_channels']
            self.layers.append(BatchNorm(in_channels))
            activation = layer_cfg.get('activation', 'ReLU')
            if activation == 'ReLU':
                self.layers.append(ReLU())
            elif activation == 'Sigmoid':
                self.layers.append(torch.nn.Sigmoid())
            elif activation == 'Tanh':
                self.layers.append(torch.nn.Tanh())
            elif activation == 'LeakyReLU':
                self.layers.append(torch.nn.LeakyReLU())
            # otherwise not adding any activation

        self.mu_layer = nn.Linear(in_channels, latent_dim if latent_dim is not None else in_channels)
        self.logvar_layer = nn.Linear(in_channels, latent_dim if latent_dim is not None else in_channels)
        self.embedding_dim = latent_dim if latent_dim is not None else in_channels

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            if isinstance(layer, (GCNConv, GATConv)):
                x = layer(x, edge_index)
            elif isinstance(layer, NNConv):
                if edge_attr is not None:
                    x = layer(x, edge_index, edge_attr)
                else:
                    num_edges = edge_index.size(1)
                    edge_attr = torch.zeros((num_edges, 0), device=x.device)  # Default edge_attr
                    x = layer(x, edge_index, edge_attr)
            elif isinstance(layer, EdgeConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        # Clamping logvar to prevent numerical issues
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        mu = torch.clamp(mu, min=-1e6, max=1e6)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, std

class VGAE(torch.nn.Module):
    def __init__(self, in_channels, cfg_layers, edge_feat_dim,
                 binary_indices, multi_class_info, multi_class_info_order, continuous_indices, ranking_indices,
                 edge_binary_indices, edge_continuous_indices, edge_multi_class_info, edge_multi_class_info_order, edge_ranking_indices,
                latent_dim=None):
        super(VGAE, self).__init__()
        self.encoder = VGAEEncoder(in_channels, cfg_layers, edge_feat_dim, latent_dim)
        last_layer_output_channels = latent_dim if latent_dim else cfg_layers[-1]['out_channels']

        self.decoder = GAEDecoder(last_layer_output_channels, binary_indices, multi_class_info, multi_class_info_order,
                                  continuous_indices, ranking_indices, edge_binary_indices, edge_continuous_indices,
                                  edge_multi_class_info, edge_multi_class_info_order, edge_ranking_indices)

    def forward(self, x, edge_index, edge_attr=None):
        z, mu, logvar = self.encoder(x, edge_index, edge_attr)
        reconstructed_x, reconstructed_adj, reconstructed_edge_features, ranking_features, edge_ranking_features = self.decoder(z, edge_index)
        return reconstructed_x, reconstructed_adj, reconstructed_edge_features, ranking_features, edge_ranking_features, mu, logvar


