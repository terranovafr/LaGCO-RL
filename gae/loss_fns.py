#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

"""
    loss_fns.py
    This file contains the loss functions used for training the GAE. The main loss is the contrastive adjacency loss, which encourages the model to learn embeddings that can distinguish between connected and non-connected nodes. Additionally, a ranking loss is implemented to ensure that the model learns to rank node pairs correctly based on their true relationships. Finally, linear and squared loss functions are provided to combine multiple loss components with configurable weights.
"""

import torch
import math
import torch.nn.functional as F

def contrastive_adj_loss(
    z,
    edge_index,
    num_negatives=1,
    tau=0.5
):
    device = z.device
    N, d = z.size()
    pos_u, pos_v = edge_index
    E = pos_u.size(0)

    # Normalize embeddings (VERY important)
    z = F.normalize(z, dim=-1)

    # ---- Positive similarities ----
    z_u = z[pos_u]                         # [E, d]
    z_v = z[pos_v]                         # [E, d]
    sim_pos = (z_u * z_v).sum(dim=-1, keepdim=True) / tau  # [E,1]

    # ---- Negative sampling ----
    neg_v = torch.empty(E, num_negatives, dtype=torch.long, device=device)

    for i in range(E):
        cnt = 0
        while cnt < num_negatives:
            candidate = torch.randint(0, N, (1,), device=device)
            if candidate != pos_v[i]:
                neg_v[i, cnt] = candidate
                cnt += 1

    z_neg = z[neg_v]                       # [E, K, d]
    sim_neg = (z_u.unsqueeze(1) * z_neg).sum(dim=-1) / tau  # [E,K]

    # ---- InfoNCE ----
    logits = torch.cat([sim_pos, sim_neg], dim=1)  # [E, 1+K]
    labels = torch.zeros(E, dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, labels)
    loss = loss / math.log(1 + 2*num_negatives)
    return loss




def ranking_loss_fun(pred, true, margin=1.0, eps=1e-8):
    # pred, true: [N, F]
    N, Feat = pred.shape

    # Pairwise differences
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # [N, N, F]
    true_diff = true.unsqueeze(1) - true.unsqueeze(0)  # [N, N, F]

    # Consider only i < j pairs (upper triangular)
    pair_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=pred.device), diagonal=1)
    pair_mask = pair_mask.unsqueeze(-1).expand(N, N, Feat)

    # Ranking masks
    pos_mask = (true_diff > 0) & pair_mask
    neg_mask = (true_diff < 0) & pair_mask

    # Hinge losses
    loss_pos = F.relu(margin - pred_diff)[pos_mask]
    loss_neg = F.relu(margin + pred_diff)[neg_mask]

    # Count valid pairs
    num_pairs = pos_mask.sum() + neg_mask.sum() + eps

    # Combine and scale
    loss = (loss_pos.sum() + loss_neg.sum()) / num_pairs
    loss = loss / margin

    return loss

def linear_loss(losses, weights):
    feature_loss = (
        weights.get('binary_cat_loss', 1.0) * losses['binary_cat_loss'] +
        weights.get('multi_cat_loss', 1.0) * losses['multi_cat_loss'] +
        weights.get('cont_loss', 1.0) * losses['cont_loss'] +
        weights.get('ranking_loss', 1.0) * losses.get('ranking_loss')
    )
    edge_feature_loss = (
        weights.get('edge_binary_loss', 1.0) * losses['edge_binary_loss'] +
        weights.get('edge_multi_cat_loss', 1.0) * losses['edge_multi_cat_loss'] +
        weights.get('edge_continuous_loss', 1.0) * losses['edge_continuous_loss'] +
        weights.get('edge_ranking_loss', 1.0) * losses.get('edge_ranking_loss', 0.0)
    )
    return (
        weights.get('adj_loss', 1.0) * losses['adj_loss'] +
        feature_loss +
        edge_feature_loss +
        weights.get('kl_loss', 1.0) * losses.get('kl_loss', 0.0),
        feature_loss,
        edge_feature_loss
    )


def squared_loss(losses, weights):
    feature_loss = (
        weights.get('binary_cat_loss', 1.0) * losses['binary_cat_loss']**2 +
        weights.get('multi_cat_loss', 1.0) * losses['multi_cat_loss']**2 +
        weights.get('cont_loss', 1.0) * losses['cont_loss']**2 +
        weights.get('ranking_loss', 1.0) * losses.get('ranking_loss')**2
    )
    edge_feature_loss = (
        weights.get('edge_binary_loss', 1.0) * losses['edge_binary_loss']**2 +
        weights.get('edge_multi_cat_loss', 1.0) * losses['edge_multi_cat_loss']**2 +
        weights.get('edge_continuous_loss', 1.0) * losses['edge_continuous_loss']**2 +
        weights.get('edge_ranking_loss', 1.0) * losses.get('edge_ranking_loss')**2
    )
    return (
        weights.get('adj_loss', 1.0) * losses['adj_loss']**2 +
        feature_loss +
        edge_feature_loss +
        weights.get('kl_loss', 1.0) * losses.get('kl_loss', 0.0)**2,
        feature_loss,
        edge_feature_loss
    )
