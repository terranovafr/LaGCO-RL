#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

"""
    gae_utils.py
    This file contains the utility functions for the Graph Autoencoder model training and validation.
"""

import torch
import torch.nn.functional as F
import os
import sys
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import DataLoader
import importlib
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
torch.set_default_dtype(torch.float32)
from model import GAE, VGAE  # noqa: E402
from loss_fns import ranking_loss_fun, contrastive_adj_loss  # noqa: E402

# Training function for the model
def compute_backward_batch_train_loss(model, graph_name, batch, loss_weighting_object, optimizer=None, backward=False, **config):
    model.train()
    if optimizer is not None:
        optimizer.zero_grad()

    total_loss, adj_loss, feature_loss, edge_feature_loss, binary_cat_loss, multi_cat_loss, cont_loss, ranking_loss, edge_binary_loss, edge_multi_cat_loss, edge_continuous_loss, edge_ranking_loss, kl_loss = compute_loss(model, graph_name, batch, loss_weighting_object, **config)
    if backward and optimizer is not None:
        total_loss.backward()
        optimizer.step()
    return total_loss.item(), adj_loss.item(), feature_loss.item(), edge_feature_loss.item(),  binary_cat_loss.item(), multi_cat_loss.item(), cont_loss.item(), ranking_loss.item(), edge_binary_loss.item(), edge_multi_cat_loss.item(), edge_continuous_loss.item(), edge_ranking_loss.item(), kl_loss.item()

# Overall loss function
def compute_loss(model, graph_name, data, loss_weighting_object, loss_combination_function,  binary_indices=None, multi_class_info=None, continuous_indices=None, ranking_indices=None, edge_binary_indices=None, edge_continuous_indices=None, edge_multi_class_info=None, edge_ranking_indices=None,  adj_weight=1,
          ranking_weight=1, edge_ranking_weight=1, binary_cat_weight=1, multi_cat_weight=1, cont_weight=1, edge_binary_cat_weight=1, edge_multi_cat_weight=1, edge_cont_weight=1, kl_weight=1,
          adj_num_negatives_per_positive=1, adj_tau=0.5, cont_loss_fun='mse', **config):

    # Determine the vector of reconstructed node features, adjacency matrix, and edge features by the model
    output = model(data.x, data.edge_index, data.edge_attr)
    device = data.x.device
    if isinstance(model, GAE):
        reconstructed_x, reconstructed_adj, reconstructed_edge_attr, ranking_features, edge_ranking_features = output
        mu, std = None, None
    else: # VGAE
        reconstructed_x, reconstructed_adj, reconstructed_edge_attr, ranking_features, edge_ranking_features, mu, std = output

    # Binary adjacency (original case)
    adj_loss = contrastive_adj_loss(
        z=reconstructed_x,  # your node embeddings
        edge_index=data.edge_index,  # positive edges
        num_negatives=adj_num_negatives_per_positive,  # optional: number of negative samples per positive edge
        tau=adj_tau  # optional: temperature
    )

    binary_cat_loss = torch.tensor(0.0, device=device)
    multi_cat_loss = torch.tensor(0.0, device=device)
    cont_loss = torch.tensor(0.0, device=device)
    ranking_loss_value = torch.tensor(0.0, device=device)
    edge_binary_loss = torch.tensor(0.0, device=device)
    edge_multi_cat_loss = torch.tensor(0.0, device=device)
    edge_continuous_loss = torch.tensor(0.0, device=device)
    edge_ranking_loss = torch.tensor(0.0, device=device)

    # Use binary cross-entropy loss for binary features
    if len(binary_indices[graph_name]) > 0:
        binary_cat_loss = F.binary_cross_entropy_with_logits(
            reconstructed_x[:, binary_indices[graph_name]], data.x[:, binary_indices[graph_name]])

    # Use multi-class cross-entropy loss for multi-class features
    offset = len(binary_indices[graph_name])
    offset_ground_truth = len(binary_indices[graph_name])
    if sum(multi_class_info[graph_name].values()) > 0:
        for idx, num_classes in multi_class_info[graph_name].items():
            logits = reconstructed_x[:, offset:offset + num_classes]
            ground_truth = data.x[:, offset_ground_truth].long()
            multi_cat_loss += F.cross_entropy(logits, ground_truth)
            offset += num_classes
            offset_ground_truth += 1
        multi_cat_loss /= len(multi_class_info[graph_name])

    # Build mapping from absolute → continuous local index
    continuous_local_map = {
        g_idx: local_idx
        for local_idx, g_idx in enumerate(continuous_indices[graph_name])
    }
    # Use mean squared error loss for continuous features
    if len(continuous_indices[graph_name]) > 0:
        cont_global = continuous_indices[graph_name]
        cont_local = [continuous_local_map[i] for i in cont_global]
        x_true = data.x[:, continuous_indices[graph_name]]
        x_pred = reconstructed_x[:, cont_local]
        if cont_loss_fun == "l1":
            cont_loss = F.l1_loss(x_pred, x_true, reduction="mean")
        elif cont_loss_fun == "mse":
            cont_loss = F.mse_loss(x_pred, x_true)

    # Use ranking loss for ranking features
    if len(ranking_indices[graph_name]) > 0:
        ranking_loss_value = ranking_loss_fun(ranking_features,
                                        data.x[:, ranking_indices[graph_name]])

    # Use binary cross-entropy loss for edge binary features
    if len(edge_binary_indices[graph_name]) > 0:
        edge_binary_loss = F.binary_cross_entropy_with_logits(
            reconstructed_edge_attr[:, edge_binary_indices[graph_name]], data.edge_attr[:, edge_binary_indices[graph_name]])

    # Use multi-class cross-entropy loss for multi-class features
    if len(edge_multi_class_info[graph_name]) > 0:
        offset = len(edge_binary_indices[graph_name])
        offset_ground_truth = len(edge_binary_indices[graph_name])
        for idx, num_classes in edge_multi_class_info[graph_name].items():
            logits = reconstructed_edge_attr[:, offset:offset + num_classes]
            ground_truth = data.edge_attr[:, offset_ground_truth].long()
            edge_multi_cat_loss += F.cross_entropy(logits, ground_truth)
            offset += num_classes
            offset_ground_truth += 1
        edge_multi_cat_loss /= len(edge_multi_class_info[graph_name])

    # Build mapping from absolute → continuous local index
    if len(edge_continuous_indices[graph_name]) > 0:
        continuous_local_map = {
            g_idx: local_idx
            for local_idx, g_idx in enumerate(edge_continuous_indices[graph_name])
        }
        # Use mean squared error loss for continuous features
        cont_global = edge_continuous_indices[graph_name]
        cont_local = [continuous_local_map[i] for i in cont_global]

        x_true = data.edge_attr[:, edge_continuous_indices[graph_name]]
        x_pred = reconstructed_edge_attr[:, cont_local]

        if cont_loss_fun == "l1":
            edge_continuous_loss = F.l1_loss(x_pred, x_true, reduction="mean")
        elif cont_loss_fun == "mse":
            edge_continuous_loss = F.mse_loss(x_pred, x_true)

    if len(edge_ranking_indices[graph_name]) > 0:
        edge_ranking_loss = ranking_loss_fun(edge_ranking_features,
                                             data.edge_attr[:, edge_ranking_indices[graph_name]])
    # Compute KL divergence for VGAE
    if isinstance(model, VGAE):
        logvar = torch.log(std ** 2 + 1e-8)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        kl_loss = torch.tensor(0.0, device=data.x.device)


    # Combine all losses with proper weights using the specified loss combination function
    loss_utils = importlib.import_module("loss_fns")
    if hasattr(loss_utils, loss_combination_function):
        loss_fn = getattr(loss_utils, loss_combination_function)
    else:
        raise ValueError(f"Loss function '{loss_combination_function}' not found in utils.loss_utils")

    current_losses = {
        'adj_loss': adj_loss,
        'binary_cat_loss': binary_cat_loss,
        'multi_cat_loss': multi_cat_loss,
        'cont_loss': cont_loss,
        'ranking_loss': ranking_loss_value,
        'edge_binary_loss': edge_binary_loss,
        'edge_multi_cat_loss': edge_multi_cat_loss,
        'edge_continuous_loss': edge_continuous_loss,
        'edge_ranking_loss': edge_ranking_loss,
        'kl_loss': kl_loss
    }
    if loss_weighting_object:
        # check if all weights are 1, this is the initial case
        loss_weighting_object[graph_name].update_losses(current_losses)
        weights = loss_weighting_object[graph_name].get_weights()
        total_loss, feature_loss, edge_feature_loss = loss_fn(current_losses, weights)

    else:
        weights = {
            'adj_loss': adj_weight,
            'binary_cat_loss': binary_cat_weight,
            'multi_cat_loss': multi_cat_weight,
            'cont_loss': cont_weight,
            'ranking_loss': ranking_weight,
            'edge_binary_loss': edge_binary_cat_weight,
            'edge_multi_cat_loss': edge_multi_cat_weight,
            'edge_continuous_loss': edge_cont_weight,
            'edge_ranking_loss': edge_ranking_weight,
            'kl_loss': kl_weight  # add KL divergence here
        }
        # Combine all losses with proper weights
        total_loss, feature_loss, edge_feature_loss = loss_fn(current_losses, weights)

    return total_loss, adj_loss, feature_loss, edge_feature_loss, binary_cat_loss, multi_cat_loss, cont_loss, ranking_loss_value, edge_binary_loss, edge_multi_cat_loss, edge_continuous_loss, edge_ranking_loss, kl_loss


# Validation function: periodically use and switch the validation set of graphs
def validate(models, val_env, loss_weighting_object, writer, config, starting_epoch, logger):
    # Set the model to evaluation mode
    for model_name in models:
        models[model_name].eval()

    total_val_loss = 0
    total_adj_loss = 0
    total_feature_loss =0
    total_edge_feature_loss = 0
    total_binary_cat_loss = 0
    total_multi_cat_loss = 0
    total_cont_loss = 0
    total_ranking_loss = 0
    total_edge_binary_loss = 0
    total_edge_multi_cat_loss = 0
    total_edge_continuous_loss = 0
    total_edge_ranking_loss = 0
    total_kl_loss = 0
    count = 0
    done = True
    batch_graphs = {}
    for iteration in range(config['val_iterations']):
        if done:
            val_env.reset()
        # Sample a valid action and advance the graph to ensure to see another configuration
        action = val_env.sample_valid_action()
        done = val_env.step(action)
        Gs = val_env.current_env.get_graphs()
        for graph_name in Gs:
            G = Gs[graph_name]
            data = from_networkx(G)
            if 'embedding' not in data: # If no edges are present (graph with just discovered nodes) add a fictious set of edges with zero embeddings
                data.edge_attr = torch.zeros((data.num_edges, config['edge_feature_vector_size'][graph_name]), device=data.x.device)
            if not graph_name in batch_graphs:
                batch_graphs[graph_name] = []
            batch_graphs[graph_name].append(data)
            if len(batch_graphs[graph_name]) % config['model_config']['batch_size'] == 0:
                batch_loader = DataLoader(batch_graphs[graph_name], batch_size=config['model_config']['batch_size'], shuffle=True)
                for batch_data in batch_loader:
                    with torch.no_grad():  # No gradients needed for validation step
                        total_loss, adj_loss, feature_loss, edge_feature_loss, binary_cat_loss, multi_cat_loss, cont_loss, ranking_loss, edge_binary_loss, edge_multi_cat_loss, edge_continuous_loss, edge_ranking_loss, kl_loss = compute_loss(models[graph_name], graph_name, batch_data, loss_weighting_object, **config)
                        total_val_loss += total_loss.item()
                        total_adj_loss += adj_loss.item()
                        total_feature_loss += feature_loss.item()
                        if not torch.isnan(edge_feature_loss).any():
                            total_edge_feature_loss += edge_feature_loss.item()
                        total_cont_loss += cont_loss.item()
                        total_binary_cat_loss += binary_cat_loss.item()
                        total_multi_cat_loss += multi_cat_loss.item()
                        total_ranking_loss += ranking_loss.item()
                        total_edge_binary_loss += edge_binary_loss.item()
                        total_edge_multi_cat_loss += edge_multi_cat_loss.item()
                        total_edge_continuous_loss += edge_continuous_loss.item()
                        total_edge_ranking_loss += edge_ranking_loss.item()
                        total_kl_loss += kl_loss.item()
                        count += 1
                batch_graphs[graph_name] = []  # Reset batch graphs for this graph name

            writer.add_scalar(f'val/{graph_name}/total_loss', total_val_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/adj_loss', total_adj_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/node_feature_vector_loss', total_feature_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/edge_feature_vector_loss', total_edge_feature_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/node_feature_vector/binary_cat_loss', total_binary_cat_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/node_feature_vector/multi_cat_loss', total_multi_cat_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/node_feature_vector/cont_loss', total_cont_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/node_feature_vector/ranking_loss', total_ranking_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/edge_feature_vector/binary_cat_loss', total_edge_binary_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/edge_feature_vector/multi_cat_loss', total_edge_multi_cat_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/edge_feature_vector/cont_loss', total_edge_continuous_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/edge_feature_vector/ranking_loss', total_edge_ranking_loss / count if count else 0, starting_epoch+config['val_iterations'])
            writer.add_scalar(f'val/{graph_name}/kl_loss', total_kl_loss / count if count else 0, starting_epoch+config['val_iterations'])


    mean_val_loss = total_val_loss / count if count else 0
    mean_adj_loss = total_adj_loss / count if count else 0
    mean_feature_loss = total_feature_loss / count if count else 0
    mean_edge_feature_loss = total_edge_feature_loss / count if count else 0
    mean_binary_cat_loss = total_binary_cat_loss / count if count else 0
    mean_multi_cat_loss = total_multi_cat_loss / count if count else 0
    mean_cont_loss = total_cont_loss / count if count else 0
    mean_ranking_loss = total_ranking_loss / count if count else 0
    mean_edge_binary_loss = total_edge_binary_loss / count if count else 0
    mean_edge_multi_cat_loss = total_edge_multi_cat_loss / count if count else 0
    mean_edge_continuous_loss = total_edge_continuous_loss / count if count else 0
    mean_edge_ranking_loss = total_edge_ranking_loss / count if count else 0
    mean_kl_loss = total_kl_loss / count if count else 0

    if config['verbose']:
        logger.info(f"Validation - Average val loss: {mean_val_loss} Average adj loss: {mean_adj_loss} Average node feature vector loss: {mean_feature_loss} Average edge feature vector loss: {mean_edge_feature_loss} Average node feature vector binary cat loss: {mean_binary_cat_loss} Average node feature vector multi cat loss: {mean_multi_cat_loss} Average node feature vector cont loss: {mean_cont_loss},"
                    f" Average node feature vector ranking loss: {mean_ranking_loss} "
                    f" Average edge feature vector binary cat loss: {mean_edge_binary_loss} Average edge feature vector multi cat loss: {mean_edge_multi_cat_loss} Average edge feature vector cont loss: {mean_edge_continuous_loss} "
                    f" Average edge feature vector ranking loss: {mean_edge_ranking_loss} "
                    f"Average KL loss: {mean_kl_loss}")

    for model_name in models:
        models[model_name].train()  # Set the model back to training mode

    return mean_val_loss