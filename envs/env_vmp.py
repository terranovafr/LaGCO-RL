#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# You may redistribute it and/or modify it under the terms of the AGPL-3.0.
# See the LICENSE file in the project root for the full license text.

'''
    env_vmp.py
    This module defines the VMPlacementEnv class, which is a reinforcement learning environment for placing Virtual Machines (VMs) onto Physical Machines (PMs) to optimize several indicators. The environment is built using the Gymnasium framework and utilizes NetworkX for graph representation of the allocation, traffic, and latency graphs.
'''

import copy
from typing import Optional
import networkx as nx
from gymnasium.spaces import Box, Dict
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from utils.features_utils import get_number_nodes, get_number_edges, get_average_node_degree, get_graph_density
from utils.model import Node, Graph, Attribute, ActionSpace, Function
from utils.feature_extractors import OneHotEncoding
from utils.pooling_functions import MeanPooling, SumPooling, MaxPooling, MinPooling
from utils.encoding_options import ContinuousValue, BinaryValue
from wrappers.wrapper import ContinuousEnv
from utils.log_utils import graph_nodes_to_text, graph_edges_to_text
from utils.math_utils import sample_range_dict, min_max_normalize
from envs.env import DiscreteEnv


class VMPlacementEnv(DiscreteEnv):
    """
        Environment for placing Virtual Machines (VMs) onto Physical Machines (PMs) to optimize several indicators.
    """

    def __init__(
            self,
            n_pms: int = 8,
            n_vms: int = 20,
            n_tenants: int = 3,
            pm_capacity_mips_min: int = 1000,
            pm_capacity_mips_max: int = 5000,
            pm_capacity_memory_min: int = 16,
            pm_capacity_memory_max: int = 128,
            pm_capacity_storage_min: int = 100,
            pm_capacity_storage_max: int = 5000,
            pm_capacity_pe_min: int = 4,
            pm_capacity_pe_max: int = 128,
            vm_demand_mips_min: int = 250,
            vm_demand_mips_max: int = 2000,
            vm_demand_memory_min: int = 2,
            vm_demand_memory_max: int = 64,
            vm_demand_storage_min: int = 10,
            vm_demand_storage_max: int = 1000,
            vm_demand_pe_min: int = 1,
            vm_demand_pe_max: int = 16,
            min_traffic: float = 1.0,
            max_traffic: float = 10.0,
            traffic_density: float = 0.2,
            latency_min: float = 1.0,
            latency_max: float = 20.0,
            p_idle: float = 60.0,
            p_peak: float = 200.0,
            same_pm_penalty_sum: float = -1.0,
            util_coefficient: float = 1,
            packing_coefficient: float = 1,
            power_coefficient: float = 1,
            latency_coefficient: float = 1,
            movement_coefficient: float = -0.05,
            add_movement_cost: bool = False,
            extra_edge_probability: float = 0.1,
            vm_vuln_prob_min: float = 0.01,
            vm_vuln_prob_max: float = 1,
            pm_escape_prob_min: float = 0.01,
            pm_escape_prob_max: float = 0.33,
            security_coefficient: float = 1.0,
            max_sweeps: int = 2500,
            invalid_action_penalty_sum: float = -10,
            revisiting: bool = False,
            alloc_graph: Optional[nx.Graph] = None,
            traffic_graph: Optional[nx.Graph] = None,
            latency_graph: Optional[nx.Graph] = None,
            scenario_graphs: Optional[dict] = None,
            **kwargs
    ):
        super().pre_init(**kwargs)
        self.n_pms = n_pms
        self.n_vms = n_vms
        self.n_tenants = n_tenants
        self.same_pm_penalty_sum = same_pm_penalty_sum
        self.metric_weights = {
            "avg_util": util_coefficient,
            "packing_efficiency": packing_coefficient,
            "avg_power": power_coefficient,
            "avg_load": latency_coefficient,
            "avg_security_risk": security_coefficient,
        }
        self.movement_coefficient = movement_coefficient
        self.add_movement_cost = add_movement_cost
        self.extra_edge_probability = extra_edge_probability
        self.vm_vuln_prob_min = vm_vuln_prob_min
        self.vm_vuln_prob_max = vm_vuln_prob_max
        self.pm_escape_prob_min = pm_escape_prob_min
        self.pm_escape_prob_max = pm_escape_prob_max
        self.max_sweeps = max_sweeps
        self.invalid_action_penalty_sum = invalid_action_penalty_sum
        self.revisiting = revisiting
        supplied_graphs = (alloc_graph is not None, traffic_graph is not None, latency_graph is not None)
        if any(supplied_graphs) and not all(supplied_graphs):
            raise ValueError("Provide all three graphs: alloc_graph, traffic_graph, and latency_graph.")
        if all(supplied_graphs):
            if scenario_graphs is not None:
                raise ValueError("Use either the three graph arguments or scenario_graphs, not both.")
            scenario_graphs = {
                "alloc_graph": alloc_graph,
                "traffic_graph": traffic_graph,
                "latency_graph": latency_graph,
            }
        self.scenario_graphs = scenario_graphs

        self.pm_capacity_ranges = {
            "pe": (pm_capacity_pe_min, pm_capacity_pe_max),
            "mips": (pm_capacity_mips_min, pm_capacity_mips_max),
            "ram": (pm_capacity_memory_min, pm_capacity_memory_max),
            "storage": (pm_capacity_storage_min, pm_capacity_storage_max)
        }
        self.vm_demand_ranges = {
            "pe": (vm_demand_pe_min, vm_demand_pe_max),
            "mips": (vm_demand_mips_min, vm_demand_mips_max),
            "ram": (vm_demand_memory_min, vm_demand_memory_max),
            "storage": (vm_demand_storage_min, vm_demand_storage_max)
        }
        self.minimize_metrics = {
            "avg_power",
            "avg_load",
            "avg_security_risk",
        }

        self.min_traffic = min_traffic
        self.max_traffic = max_traffic
        self.traffic_density = traffic_density
        self.latency_min, self.latency_max = latency_min, latency_max
        self.p_idle = p_idle
        self.p_peak = p_peak

        # Graphs
        self.alloc_graph = nx.Graph()  # VM-PM allocation graph
        self.traffic_graph = nx.Graph()  # VM-VM traffic graph (static)

        # Gym API
        self.pm_feat_dim = 8
        self.vm_feat_dim = 7
        self.obs_dim = self.n_pms * self.pm_feat_dim + self.n_vms * self.vm_feat_dim

        self.observation_space = Dict({
            "graph": Box(
                low=0,
                high=1,
                shape=(self.obs_dim,),
                dtype=np.float32
            )
        })

        # Either generate a random scenario or load the three supplied graphs.
        if self.scenario_graphs is None:
            success = False
            attempts = 0
            while not success:
                self._generate_datacenter()
                self._generate_vms()
                success = self._initialize_empirical_worst_allocation(
                    max_sweeps=self.max_sweeps
                )
                attempts += 1
                if attempts >= 5 and not success:
                    raise ValueError(
                        "Failed to initialize empirical worst allocation after 5 attempts."
                    )
        else:
            self._load_scenario_graphs(self.scenario_graphs)
            success = self._initialize_empirical_worst_allocation(
                max_sweeps=self.max_sweeps
            )
            if not success:
                raise ValueError(
                    "Could not initialize a feasible allocation from scenario_graphs. "
                    "Check PM capacities, VM demands, and graph attributes."
                )

        # Graph-based scenarios may have different sizes than the default parameters.
        self.pm_feat_dim = 8
        self.vm_feat_dim = 7
        self.obs_dim = self.n_pms * self.pm_feat_dim + self.n_vms * self.vm_feat_dim
        self.observation_space = Dict({
            "graph": Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)
        })

        # update graphs
        for vm_id in range(self.n_vms):
            _, vm_feature = self._vm_features(f"VM_{vm_id}")
            for key in vm_feature:
                self.alloc_graph.nodes[f"VM_{vm_id}"]["x_dict"][key] = vm_feature[key]
                self.traffic_graph.nodes[f"VM_{vm_id}"]["x_dict"][key] = vm_feature[key]
            self.alloc_graph.nodes[f"VM_{vm_id}"]["x_dict"]["visited"] = False

        for pm_id in range(self.n_pms):
            _, pm_feature = self._pm_features(f"PM_{pm_id}")
            for key in pm_feature:
                self.alloc_graph.nodes[f"PM_{pm_id}"]["x_dict"][key] = pm_feature[key]
                self.latency_graph.nodes[f"PM_{pm_id}"]["x_dict"][key] = pm_feature[key]
        self.initial_alloc_graph = copy.deepcopy(self.alloc_graph)
        self.action_list = self.build_action_list()
        self.semantic_action_list = self.build_action_list()
        self.scenario_size = self.n_pms + self.n_vms
        self.reset()
        super().post_init()

    @staticmethod
    def _edge_value(data, key, default=None):
        """Read an edge value from either a direct attribute or edge_attr_dict."""
        if key in data:
            return data[key]
        return data.get("edge_attr_dict", {}).get(key, default)

    def _load_scenario_graphs(self, scenario_graphs):
        """Load canonical alloc/traffic/latency NetworkX graphs.

        Required node labels are VM_0..VM_n and PM_0..PM_n. Node data must
        contain x_dict attributes in the same schema used by this environment.
        Traffic edges require traffic weights; latency edges require latency
        weights. The latency graph must connect all PMs.
        """
        required = {"alloc_graph", "traffic_graph", "latency_graph"}
        if not isinstance(scenario_graphs, dict) or not required.issubset(scenario_graphs):
            raise ValueError(
                "scenario_graphs must be a dict containing alloc_graph, "
                "traffic_graph, and latency_graph."
            )
        for name in required:
            graph = scenario_graphs[name]
            if not isinstance(graph, nx.Graph):
                raise TypeError(f"scenario_graphs['{name}'] must be a NetworkX graph.")

        self.alloc_graph = copy.deepcopy(scenario_graphs["alloc_graph"])
        self.traffic_graph = copy.deepcopy(scenario_graphs["traffic_graph"])
        self.latency_graph = copy.deepcopy(scenario_graphs["latency_graph"])

        pms = sorted(
            [n for n, d in self.alloc_graph.nodes(data=True)
             if d.get("x_dict", {}).get("type") == "PM"],
            key=lambda x: int(str(x).split("_")[-1]) if str(x).split("_")[-1].isdigit() else str(x),
        )
        vms = sorted(
            [n for n, d in self.alloc_graph.nodes(data=True)
             if d.get("x_dict", {}).get("type") == "VM"],
            key=lambda x: int(str(x).split("_")[-1]) if str(x).split("_")[-1].isdigit() else str(x),
        )
        expected_pms = [f"PM_{i}" for i in range(len(pms))]
        expected_vms = [f"VM_{i}" for i in range(len(vms))]
        if pms != expected_pms or vms != expected_vms:
            raise ValueError(
                "Allocation graph nodes must use contiguous labels PM_0..PM_n and VM_0..VM_n, "
                "with x_dict['type'] set to 'PM' or 'VM'."
            )
        self.n_pms, self.n_vms = len(pms), len(vms)
        if self.n_pms < 2 or self.n_vms < 2:
            raise ValueError("A scenario needs at least two PMs and two VMs.")

        # Validate and infer resource ranges from supplied attributes.
        for pm in pms:
            info = self.alloc_graph.nodes[pm].get("x_dict", {})
            for key in ("cap", "used", "escape_prob"):
                if key not in info:
                    raise ValueError(f"{pm} is missing x_dict[{key!r}].")
            for resource in ("pe", "mips", "ram", "storage"):
                if resource not in info["cap"]:
                    raise ValueError(f"{pm} capacity is missing resource {resource!r}.")
        for vm in vms:
            info = self.alloc_graph.nodes[vm].get("x_dict", {})
            for key in ("dem", "tenant", "vuln_prob"):
                if key not in info:
                    raise ValueError(f"{vm} is missing x_dict[{key!r}].")
            for resource in ("pe", "mips", "ram", "storage"):
                if resource not in info["dem"]:
                    raise ValueError(f"{vm} demand is missing resource {resource!r}.")

        resources = ("pe", "mips", "ram", "storage")
        self.pm_capacity_ranges = {
            r: (min(self.alloc_graph.nodes[p]["x_dict"]["cap"][r] for p in pms),
                max(self.alloc_graph.nodes[p]["x_dict"]["cap"][r] for p in pms))
            for r in resources
        }
        self.vm_demand_ranges = {
            r: (min(self.alloc_graph.nodes[v]["x_dict"]["dem"][r] for v in vms),
                max(self.alloc_graph.nodes[v]["x_dict"]["dem"][r] for v in vms))
            for r in resources
        }
        # Avoid division by zero in feature normalization for constant-valued resources.
        for ranges in (self.pm_capacity_ranges, self.vm_demand_ranges):
            for resource, (lo, hi) in list(ranges.items()):
                if hi == lo:
                    ranges[resource] = (lo - 0.5, hi + 0.5)
        tenants = [self.alloc_graph.nodes[v]["x_dict"]["tenant"] for v in vms]
        self.n_tenants = max(tenants, default=0) + 1
        self.vm_vuln_prob_min = min(self.alloc_graph.nodes[v]["x_dict"]["vuln_prob"] for v in vms)
        self.vm_vuln_prob_max = max(self.alloc_graph.nodes[v]["x_dict"]["vuln_prob"] for v in vms)
        self.pm_escape_prob_min = min(self.alloc_graph.nodes[p]["x_dict"]["escape_prob"] for p in pms)
        self.pm_escape_prob_max = max(self.alloc_graph.nodes[p]["x_dict"]["escape_prob"] for p in pms)

        # Ensure the two static graphs have the expected VM/PM nodes and features.
        if set(self.traffic_graph.nodes) != set(vms):
            raise ValueError("traffic_graph must contain exactly the VM nodes in alloc_graph.")
        if set(self.latency_graph.nodes) != set(pms):
            raise ValueError("latency_graph must contain exactly the PM nodes in alloc_graph.")
        for vm in vms:
            self.traffic_graph.nodes[vm].setdefault("x_dict", {})
            self.traffic_graph.nodes[vm]["x_dict"].update(
                copy.deepcopy(self.alloc_graph.nodes[vm]["x_dict"])
            )
        for pm in pms:
            self.latency_graph.nodes[pm].setdefault("x_dict", {})
            self.latency_graph.nodes[pm]["x_dict"].update(
                copy.deepcopy(self.alloc_graph.nodes[pm]["x_dict"])
            )

        for u, v, data in self.traffic_graph.edges(data=True):
            if self._edge_value(data, "traffic") is None:
                raise ValueError(f"Traffic edge ({u}, {v}) has no traffic weight.")
        for u, v, data in self.latency_graph.edges(data=True):
            latency = self._edge_value(data, "latency")
            if latency is None or float(latency) < 0:
                raise ValueError(f"Latency edge ({u}, {v}) needs a non-negative latency weight.")
            data["weight"] = float(latency)
        if not nx.is_connected(self.latency_graph):
            raise ValueError("latency_graph must be connected so every PM pair has a path.")

        self.latency_min = min((self._edge_value(d, "latency") for _, _, d in self.latency_graph.edges(data=True)), default=1.0)
        self.latency_max = max((self._edge_value(d, "latency") for _, _, d in self.latency_graph.edges(data=True)), default=20.0)
        self.min_traffic = min((self._edge_value(d, "traffic") for _, _, d in self.traffic_graph.edges(data=True)), default=1.0)
        self.max_traffic = max((self._edge_value(d, "traffic") for _, _, d in self.traffic_graph.edges(data=True)), default=10.0)
        self.p_idle = float(np.mean([self.alloc_graph.nodes[p]["x_dict"].get("power", 60.0) for p in pms]))
        self.p_peak = max(self.p_idle + 1.0, float(self.p_peak))
        self.latency_dict = {u: dict(row) for u, row in nx.floyd_warshall(self.latency_graph, weight="weight").items()}
        self.max_latency_movement_cost = max(
            (self.latency_dict[u][v] for u in pms for v in pms if u != v),
            default=1.0,
        )

        # Initialize usage and power from the supplied capacity/demand data;
        # the empirical initializer will construct a feasible allocation.
        for pm in pms:
            self.alloc_graph.nodes[pm]["x_dict"]["used"] = {r: 0 for r in resources}
            self.alloc_graph.nodes[pm]["x_dict"]["power"] = self.p_idle
        self.alloc_graph.remove_edges_from(list(self.alloc_graph.edges()))
        self.initial_alloc_graph = copy.deepcopy(self.alloc_graph)

    def _record_metrics_history(self):
        """Append a numeric snapshot of objectives and reward for plotting."""
        if not hasattr(self, "metrics_history"):
            self.metrics_history = []
        row = {"step": int(getattr(self, "current_step", getattr(self, "steps", 0))),
               "reward": float(getattr(self, "reward", 0.0))}
        for key, value in getattr(self, "current_metrics", {}).items():
            if np.isscalar(value):
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
        self.metrics_history.append(row)

    def render(self, show=True, save_path=None):
        """Render VM allocation and objective history."""
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.flatten()

        # 1) Current VM allocation.
        graph = self.alloc_graph
        pos = nx.spring_layout(graph, seed=7)

        pm_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("x_dict", {}).get("type") == "PM"
        ]
        vm_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("x_dict", {}).get("type") == "VM"
        ]

        nx.draw_networkx_edges(
            graph, pos, ax=axes[0], edge_color="#BBBBBB", width=1
        )
        nx.draw_networkx_nodes(
            graph, pos, nodelist=pm_nodes, node_color="#4C78A8",
            node_shape="s", node_size=250, ax=axes[0], label="PM"
        )
        nx.draw_networkx_nodes(
            graph, pos, nodelist=vm_nodes, node_color="#F58518",
            node_size=100, ax=axes[0], label="VM"
        )

        if graph.number_of_nodes() <= 35:
            nx.draw_networkx_labels(graph, pos, ax=axes[0], font_size=6)

        axes[0].set_title("VM allocation")
        axes[0].axis("off")
        axes[0].legend(fontsize=7, loc="best")

        # 2) Objective and reward history.
        history = getattr(self, "metrics_history", [])

        plots = [
            ("avg_util", "Average utilization"),
            ("packing_efficiency", "Packing efficiency"),
            ("avg_power", "Average power"),
            ("avg_load", "Communication load"),
            ("avg_security_risk", "Security risk"),
            ("weighted_sum", "Weighted objective"),
            ("reward", "Reward per step"),
        ]

        for ax, (key, title) in zip(axes[1:], plots):
            rows = [r for r in history if key in r]

            if rows:
                xs = [r["step"] for r in rows]
                ys = [r[key] for r in rows]
                ax.plot(xs, ys, linewidth=1.5)
                ax.scatter(xs, ys, s=9)

            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Step", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

        fig.suptitle(
            f"VM Placement — step "
            f"{getattr(self, 'current_step', getattr(self, 'steps', 0))}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path is not None:
            from pathlib import Path
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        if show:
            plt.show(block=False)
            plt.pause(0.001)

        return fig

    def build_action_list(self):
        action_list = []
        for vm_idx in range(self.n_vms):
            for pm_idx in range(self.n_pms):
                action_list.append((vm_idx, pm_idx))
        return action_list

    def update_config(self, config):
        super().update_config(config)
        self.same_pm_penalty_sum = config.get("same_pm_penalty_sum", self.same_pm_penalty_sum)
        self.util_coefficient = config.get("util_coefficient", self.metric_weights["avg_util"])
        self.packing_coefficient = config.get("packing_coefficient", self.metric_weights["packing_efficiency"])
        self.power_coefficient = config.get("power_coefficient", self.metric_weights["avg_power"])
        self.latency_coefficient = config.get("latency_coefficient", self.metric_weights["avg_load"])
        self.security_coefficient = config.get("security_coefficient", self.metric_weights["avg_security_risk"])
        self.movement_coefficient = config.get("movement_coefficient", self.movement_coefficient)
        self.invalid_action_penalty_sum = config.get("invalid_action_penalty_sum", self.invalid_action_penalty_sum)
        self.add_movement_cost = config.get("add_movement_cost", self.add_movement_cost)
        self.revisiting = config.get("revisiting", self.revisiting if hasattr(self, "revisiting") else False)

        # update visited flag to all False
        for vm_id in range(self.n_vms):
            self.alloc_graph.nodes[f"VM_{vm_id}"]["x_dict"]["visited"] = False
        self.action_list = self.build_action_list()
        self.semantic_action_list = self.build_action_list()
        self.post_init()
        # no need to recompute action space

    def _generate_datacenter(self):
        # Create PM nodes with random capacities and escape probabilities
        self.alloc_graph.clear()
        for i in range(self.n_pms):
            caps = {k: sample_range_dict(self.pm_capacity_ranges, k) for k in self.pm_capacity_ranges}
            escape_prob = random.uniform(self.pm_escape_prob_min, self.pm_escape_prob_max)

            self.alloc_graph.add_node(
                f"PM_{i}",
                x_dict={"type": "PM", "cap": caps, "used": {k: 0 for k in caps}, "power": self.p_idle,
                        "escape_prob": escape_prob},
            )
        # Random symmetric latencies
        self.latency_graph, self.latency_dict = self._build_connected_latency_graph(self.n_pms, self.latency_min,
                                                                                    self.latency_max,
                                                                                    extra_edge_prob=self.extra_edge_probability)

    def _build_connected_latency_graph(self, n_pms, latency_min=1, latency_max=20, extra_edge_prob=0.1):
        # Step 1: fully-defined latency matrix (NOT edges yet)
        lat_matrix = np.zeros((n_pms, n_pms))
        for i in range(n_pms):
            for j in range(i + 1, n_pms):
                lat = random.uniform(latency_min, latency_max)
                lat_matrix[i, j] = lat_matrix[j, i] = lat

        # Step 2: build a complete graph *temporarily*
        full_g = nx.complete_graph(n_pms)
        for u, v in full_g.edges():
            full_g[u][v]['weight'] = lat_matrix[u, v]

        # Step 3: compute MST to guarantee connectivity
        mst = nx.minimum_spanning_tree(full_g)

        # Step 4: convert to final sparse graph
        G = nx.Graph()
        for u, v, data in mst.edges(data=True):
            G.add_edge(f"PM_{u}", f"PM_{v}", edge_attr_dict={"latency": data['weight']})

        # Step 5: add a few random edges (sparse)
        for i in range(n_pms):
            for j in range(i + 1, n_pms):
                if not mst.has_edge(i, j) and random.random() < extra_edge_prob:
                    G.add_edge(f"PM_{i}", f"PM_{j}", edge_attr_dict={"latency": lat_matrix[i, j]})

        for _, _, data in G.edges(data=True):
            data["weight"] = data["edge_attr_dict"]["latency"]
        fw = nx.floyd_warshall(G, weight="weight")
        latency_dict = {u: dict(v) for u, v in fw.items()}
        self.max_latency_movement_cost = max(
            latency_dict[u][v] for u in latency_dict for v in latency_dict[u] if u != v)
        for node in G.nodes:
            G.nodes[node]['x_dict'] = self.alloc_graph.nodes[node]['x_dict']
        return G, latency_dict

    def _generate_vms(self):
        # Create VM nodes with random demands, tenants, and vulnerability probabilities
        self.traffic_graph.clear()
        for i in range(self.n_vms):
            dem = {k: sample_range_dict(self.vm_demand_ranges, k) for k in self.vm_demand_ranges}
            tenant = random.randint(0, self.n_tenants - 1)
            vuln_prob = random.uniform(self.vm_vuln_prob_min, self.vm_vuln_prob_max)

            # Add VM node to both graphs
            self.alloc_graph.add_node(f"VM_{i}",
                                      x_dict={"type": "VM", "dem": dem, "tenant": tenant, "vuln_prob": vuln_prob})
            self.traffic_graph.add_node(f"VM_{i}",
                                        x_dict={"type": "VM", "dem": dem, "tenant": tenant, "vuln_prob": vuln_prob})

        # Generate traffic edges in traffic graph
        for i in range(self.n_vms):
            for j in range(i + 1, self.n_vms):
                t_i = self.traffic_graph.nodes[f"VM_{i}"]["x_dict"]["tenant"]
                t_j = self.traffic_graph.nodes[f"VM_{j}"]["x_dict"]["tenant"]
                if t_i != t_j:
                    continue
                if random.random() < self.traffic_density:
                    w = random.uniform(self.min_traffic, self.max_traffic)
                    self.traffic_graph.add_edge(f"VM_{i}", f"VM_{j}", edge_attr_dict={"traffic": w})
        # if no edge added, add at least one to avoid GNN issues
        if self.traffic_graph.number_of_edges() == 0:
            # find VMs with same tenant
            tenant_to_vms = {}
            for i in range(self.n_vms):
                tenant = self.traffic_graph.nodes[f"VM_{i}"]["x_dict"]["tenant"]
                tenant_to_vms.setdefault(tenant, []).append(i)
            # randomly connect two VMs from the largest tenant group
            largest_tenant = max(tenant_to_vms, key=lambda t: len(tenant_to_vms[t]))
            vms = tenant_to_vms[largest_tenant]
            if len(vms) >= 2:
                v1, v2 = random.sample(vms, 2)
                self.traffic_graph.add_edge(f"VM_{v1}", f"VM_{v2}", edge_attr_dict={
                    "traffic": random.uniform(self.min_traffic, self.max_traffic)})
            else:
                # If even the largest tenant has only 1 VM, just connect two random VMs
                v1, v2 = random.sample(range(self.n_vms), 2)
                self.traffic_graph.add_edge(f"VM_{v1}", f"VM_{v2}", edge_attr_dict={
                    "traffic": random.uniform(self.min_traffic, self.max_traffic)})

    def _encode_allocation(self):
        # Returns a tuple of PM indices indexed by VM index
        alloc = [None] * self.n_vms
        for vm in range(self.n_vms):
            vm_node = f"VM_{vm}"
            neighbors = list(self.alloc_graph.neighbors(vm_node))
            if len(neighbors) != 1:
                raise RuntimeError("Invalid allocation encoding")
            alloc[vm] = int(neighbors[0].split("_")[1])
        return tuple(alloc)

    def _apply_allocation(self, allocation):
        # Apply VM→PM mapping given by allocation tuple

        # Reset PM usage and edges
        for i in range(self.n_pms):
            pm = f"PM_{i}"
            self.alloc_graph.nodes[pm]["x_dict"]["used"] = {k: 0 for k in self.pm_capacity_ranges}
            self.alloc_graph.nodes[pm]["x_dict"]["power"] = self.p_idle

        self.alloc_graph.remove_edges_from(list(self.alloc_graph.edges()))

        # Place VMs
        for vm_idx, pm_idx in enumerate(allocation):
            if not self.is_valid_action(f"VM_{vm_idx}", f"PM_{pm_idx}"):
                raise RuntimeError("Invalid allocation encountered")
            self._place_vm(f"VM_{vm_idx}", f"PM_{pm_idx}")

    def _initialize_empirical_worst_allocation(self, max_sweeps=500):
        # Heuristic-driven search to find a diverse set of allocations that span a wide range of metric values, including near-worst cases.

        metric_names = [
            "avg_util",
            "packing_efficiency",
            "avg_power",
            "avg_load",
            "avg_security_risk",
        ]

        self.metric_bounds = {m: [np.inf, -np.inf] for m in metric_names}
        self.empirical_allocations = []

        seen_allocs = set()
        best_score = -np.inf
        worst_score = np.inf
        worst_alloc = None

        local_steps = 3
        top_k = 3
        candidate_trials = 6



        def reset_empty_allocation():
            for i in range(self.n_pms):
                pm = f"PM_{i}"
                self.alloc_graph.nodes[pm]["x_dict"]["used"] = {k: 0 for k in self.pm_capacity_ranges}
                self.alloc_graph.nodes[pm]["x_dict"]["power"] = self.p_idle
            self.alloc_graph.remove_edges_from(list(self.alloc_graph.edges()))

        def vm_score(vm_idx):
            vm_name = f"VM_{vm_idx}"
            dem = self.alloc_graph.nodes[vm_name]["x_dict"]["dem"]

            demand_score = 0.0
            for k in self.vm_demand_ranges:
                lo, hi = self.vm_demand_ranges[k]
                demand_score += (dem[k] - lo) / (hi - lo + 1e-8)

            traffic_score = 0.0
            for nbr in self.traffic_graph.neighbors(vm_name):
                traffic_score += self._edge_value(self.traffic_graph[vm_name][nbr], "traffic", 0.0)

            return demand_score + 0.25 * traffic_score

        def projected_ratio(vm_idx, pm_idx):
            vm_name = f"VM_{vm_idx}"
            pm_name = f"PM_{pm_idx}"

            dem = self.alloc_graph.nodes[vm_name]["x_dict"]["dem"]
            used = self.alloc_graph.nodes[pm_name]["x_dict"]["used"]
            cap = self.alloc_graph.nodes[pm_name]["x_dict"]["cap"]

            return np.mean([(used[k] + dem[k]) / (cap[k] + 1e-8) for k in cap])

        def construct_allocation(direction):
            """
            direction='worst' -> seek low avg_util (balanced PMs around mid-load)
            direction='best'  -> seek high avg_util (pack VMs onto active PMs)
            """
            for _ in range(10):
                reset_empty_allocation()

                vm_order = list(range(self.n_vms))
                vm_order.sort(key=lambda vm: vm_score(vm) + random.random() * 0.25, reverse=True)

                success = True
                for vm_idx in vm_order:
                    feasible_pms = [pm_idx for pm_idx in range(self.n_pms) if self.is_valid_action(vm_idx, pm_idx)]
                    if not feasible_pms:
                        success = False
                        break

                    scored = []
                    for pm_idx in feasible_pms:
                        pm_name = f"PM_{pm_idx}"
                        used = self.alloc_graph.nodes[pm_name]["x_dict"]["used"]
                        is_active = int(any(v > 0 for v in used.values()))
                        r = projected_ratio(vm_idx, pm_idx)

                        if direction == "worst":
                            # avg_util is highest when PMs are empty/full, lowest around half-full
                            score = -abs(r - 0.5) + 0.10 * (1 - is_active)
                        else:
                            score = r + 0.20 * is_active

                        score += random.random() * 0.05
                        scored.append((pm_idx, score))

                    scored.sort(key=lambda x: x[1], reverse=True)
                    chosen_pm = random.choice(scored[:min(top_k, len(scored))])[0]
                    self._place_vm(f"VM_{vm_idx}", f"PM_{chosen_pm}")

                if success:
                    return self._encode_allocation()

            raise RuntimeError("Failed to construct allocation")

        def refine_allocation(allocation, direction):
            # Few stochastic one-VM moves: direction='worst' -> prefer lower avg_util / direction='best'  -> prefer higher avg_util
            current_alloc = allocation

            for _ in range(local_steps):
                self._apply_allocation(current_alloc)

                vm_weights = [vm_score(vm_idx) + 1e-6 for vm_idx in range(self.n_vms)]
                candidates = []

                for _ in range(candidate_trials):
                    vm_idx = random.choices(range(self.n_vms), weights=vm_weights, k=1)[0]
                    current_pm = current_alloc[vm_idx]

                    pm_candidates = list(range(self.n_pms))
                    random.shuffle(pm_candidates)

                    for pm_idx in pm_candidates:
                        if pm_idx == current_pm:
                            continue

                        candidate = list(current_alloc)
                        candidate[vm_idx] = pm_idx
                        candidate = tuple(candidate)

                        try:
                            self._apply_allocation(candidate)
                        except RuntimeError:
                            continue

                        value = self._compute_metrics(compute_norm=False)["avg_util"]
                        candidates.append((candidate, value))
                        break

                if not candidates:
                    self._apply_allocation(current_alloc)
                    return current_alloc

                vals = np.array([v for _, v in candidates], dtype=float)
                if direction == "best":
                    weights = vals - vals.min() + 1e-8
                else:
                    weights = vals.max() - vals + 1e-8

                idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
                current_alloc = candidates[idx][0]

            self._apply_allocation(current_alloc)
            return current_alloc

        for sweep in tqdm(range(max_sweeps), desc="Empirical worst allocation sweep"):
            try:
                mode = sweep % 3
                if mode == 0:
                    alloc = construct_allocation(direction="worst")
                    alloc = refine_allocation(alloc, direction="worst")
                elif mode == 1:
                    alloc = construct_allocation(direction="best")
                    alloc = refine_allocation(alloc, direction="best")
                else:
                    self._random_allocation()
                    alloc = self._encode_allocation()
                    alloc = refine_allocation(
                        alloc,
                        direction="worst" if random.random() < 0.5 else "best"
                    )
            except RuntimeError:
                continue

            if alloc in seen_allocs:
                continue

            seen_allocs.add(alloc)
            self._apply_allocation(alloc)
            metrics = self._compute_metrics(compute_norm=False)

            for m in metric_names:
                val = metrics[m]
                self.metric_bounds[m][0] = min(self.metric_bounds[m][0], val)
                self.metric_bounds[m][1] = max(self.metric_bounds[m][1], val)

            self.empirical_allocations.append({
                "alloc": alloc,
                "metrics": {m: metrics[m] for m in metric_names}
            })

        if not self.empirical_allocations:
            raise ValueError("Failed to find valid empirical allocations")

        for entry in self.empirical_allocations:
            metrics = entry["metrics"]
            score = self.compute_weighted_sum_normalized_metrics(metrics)
            entry["score"] = score

            if score > best_score:
                best_score = score

            if score < worst_score:
                worst_score = score
                worst_alloc = entry["alloc"]

        self.best_score = best_score
        self.worst_score = worst_score

        self._apply_allocation(worst_alloc)

        print(f"Unique allocations evaluated: {len(seen_allocs)}")
        print("Metrics bounds minimums:", {m: self.metric_bounds[m][0] for m in metric_names})
        print("Metrics bounds maximums:", {m: self.metric_bounds[m][1] for m in metric_names})

        return self.best_score != self.worst_score

    def compute_weighted_sum_normalized_metrics(self, metrics):
        # Compute weighted sum of normalized metrics given raw metrics dict

        eps = 1e-8
        norm_metrics = {}

        for m in ["avg_util"]:

            v = metrics[m]
            min_v, max_v = self.metric_bounds[m]

            # Min-max normalization
            norm = min_max_normalize(v, min_v, max_v, eps=eps)
            # Invert if lower-is-better
            if m in self.minimize_metrics:
                norm = 1.0 - norm

            norm_metrics[m] = norm

        # Weighted sum on NORMALIZED metrics
        score = sum(
            self.metric_weights[m] * norm_metrics[m]
            for m in ["avg_util"]
        )
        return score

    def _get_pm_of_vm(self, vm_idx):
        # Returns PM index if VM is allocated, else None
        for nbr in self.alloc_graph.neighbors(vm_idx):
            if self.alloc_graph.nodes[nbr]["x_dict"]["type"] == "PM":
                return nbr
        return None

    def is_valid_action(self, vm_idx, pm_idx) -> bool:
        vm_idx = f"VM_{vm_idx}" if isinstance(vm_idx, int) else vm_idx
        pm_idx = f"PM_{pm_idx}" if isinstance(pm_idx, int) else pm_idx
        # Check if placing VM on PM would violate capacity constraints
        if not self.revisiting:
            if self.alloc_graph.nodes[vm_idx]["x_dict"].get("visited", False):
                return False
        dem = self.alloc_graph.nodes[vm_idx]["x_dict"]["dem"]
        used = self.alloc_graph.nodes[pm_idx]["x_dict"]["used"]
        cap = self.alloc_graph.nodes[pm_idx]["x_dict"]["cap"]
        return all(used[k] + dem[k] <= cap[k] for k in cap)

    def _place_vm(self, vm_idx, pm_idx):
        # place VM on PM, update used resources and power, and return movement cost if applicable
        dem = self.alloc_graph.nodes[vm_idx]["x_dict"]["dem"]

        # instead of reinit, we can check if the VM is already placed on the target PM
        old_pm = self._get_pm_of_vm(vm_idx)
        movement_cost = 0.0
        # Remove old allocation
        pms_changed = [pm_idx]

        if old_pm is not None:
            movement_cost = self.latency_dict[old_pm][pm_idx]

            old_used = self.alloc_graph.nodes[old_pm]["x_dict"]["used"]
            for k in dem:
                old_used[k] -= dem[k]

            self.alloc_graph.remove_edge(vm_idx, old_pm)
            self.alloc_graph.nodes[old_pm]["x_dict"]["power"] = self._compute_power(old_pm)
            pms_changed.append(old_pm)

        # Allocate on new PM
        used = self.alloc_graph.nodes[pm_idx]["x_dict"]["used"]
        for k in dem:
            used[k] += dem[k]

        self.alloc_graph.add_edge(vm_idx, pm_idx)
        self.alloc_graph.nodes[pm_idx]["x_dict"]["power"] = self._compute_power(pm_idx)

        return movement_cost, [vm_idx], pms_changed

    def _compute_power(self, pm_name):
        # simple linear power model based on CPU utilization
        info = self.alloc_graph.nodes[pm_name]["x_dict"]
        utilization = np.mean([info["used"][k] / info["cap"][k] for k in info["cap"]])
        return self.p_idle + (self.p_peak - self.p_idle) * utilization

    def _pm_features(self, idx):
        info = self.alloc_graph.nodes[idx]["x_dict"]
        cap, used, power = info["cap"], info["used"], self._compute_power(idx)
        # use latency matrix to compute average latency to other PMs
        avg_latency = np.mean([self.latency_dict[idx][f"PM_{j}"] for j in range(self.n_pms) if f"PM_{j}" != idx])

        features = {
            'pe_norm_ratio': (cap["pe"] - self.pm_capacity_ranges["pe"][0]) / (
                        self.pm_capacity_ranges["pe"][1] - self.pm_capacity_ranges["pe"][0]),
            'mips_norm_ratio': (cap["mips"] - self.pm_capacity_ranges["mips"][0]) / (
                        self.pm_capacity_ranges["mips"][1] - self.pm_capacity_ranges["mips"][0]),
            'ram_norm_ratio': (cap["ram"] - self.pm_capacity_ranges["ram"][0]) / (
                        self.pm_capacity_ranges["ram"][1] - self.pm_capacity_ranges["ram"][0]),
            'storage_norm_ratio': (cap["storage"] - self.pm_capacity_ranges["storage"][0]) / (
                        self.pm_capacity_ranges["storage"][1] - self.pm_capacity_ranges["storage"][0]),
            'pe_ratio': used["pe"] / cap["pe"],
            'mips_ratio': used["mips"] / cap["mips"],
            'ram_ratio': used["ram"] / cap["ram"],
            'storage_ratio': used["storage"] / cap["storage"],
            'power_ratio': power / self.p_peak,
            'avg_latency': avg_latency / self.latency_max,
        }
        graph_features = {
            'pe_norm_ratio': (cap["pe"] - self.pm_capacity_ranges["pe"][0]) / (
                    self.pm_capacity_ranges["pe"][1] - self.pm_capacity_ranges["pe"][0]),
            'mips_norm_ratio': (cap["mips"] - self.pm_capacity_ranges["mips"][0]) / (
                    self.pm_capacity_ranges["mips"][1] - self.pm_capacity_ranges["mips"][0]),
            'ram_norm_ratio': (cap["ram"] - self.pm_capacity_ranges["ram"][0]) / (
                    self.pm_capacity_ranges["ram"][1] - self.pm_capacity_ranges["ram"][0]),
            'storage_norm_ratio': (cap["storage"] - self.pm_capacity_ranges["storage"][0]) / (
                    self.pm_capacity_ranges["storage"][1] - self.pm_capacity_ranges["storage"][0]),
            'pe_ratio': used["pe"] / cap["pe"],
            'mips_ratio': used["mips"] / cap["mips"],
            'ram_ratio': used["ram"] / cap["ram"],
            'storage_ratio': used["storage"] / cap["storage"],
            'power_ratio': power / self.p_peak,
        }
        return features, graph_features

    def _vm_features(self, idx):
        info = self.alloc_graph.nodes[idx]["x_dict"]
        dem = info["dem"]
        total_traffic = sum(self._edge_value(self.traffic_graph[idx][nbr], "traffic", 0.0)
                            for nbr in self.traffic_graph.neighbors(idx))
        pm_self = self._get_pm_of_vm(idx)
        colocated_ratio = sum(1 for nbr in self.alloc_graph.neighbors(idx)
                              if nbr.startswith("VM_") and
                              self._get_pm_of_vm(nbr) == pm_self) / max(len(list(self.alloc_graph.neighbors(idx))), 1)
        features = {
            'pe_norm_ratio': (dem["pe"] - self.vm_demand_ranges["pe"][0]) / (
                        self.vm_demand_ranges["pe"][1] - self.vm_demand_ranges["pe"][0]),
            'mips_norm_ratio': (dem["mips"] - self.vm_demand_ranges["mips"][0]) / (
                        self.vm_demand_ranges["mips"][1] - self.vm_demand_ranges["mips"][0]),
            'ram_norm_ratio': (dem["ram"] - self.vm_demand_ranges["ram"][0]) / (
                        self.vm_demand_ranges["ram"][1] - self.vm_demand_ranges["ram"][0]),
            'storage_norm_ratio': (dem["storage"] - self.vm_demand_ranges["storage"][0]) / (
                        self.vm_demand_ranges["storage"][1] - self.vm_demand_ranges["storage"][0]),
            'total_traffic': total_traffic,
            'colocated_ratio': colocated_ratio,
            'visited': int(self.alloc_graph.nodes[idx]["x_dict"].get("visited", False)),
        }
        graph_features = {
            'pe_norm_ratio': (dem["pe"] - self.vm_demand_ranges["pe"][0]) / (
                    self.vm_demand_ranges["pe"][1] - self.vm_demand_ranges["pe"][0]),
            'mips_norm_ratio': (dem["mips"] - self.vm_demand_ranges["mips"][0]) / (
                    self.vm_demand_ranges["mips"][1] - self.vm_demand_ranges["mips"][0]),
            'ram_norm_ratio': (dem["ram"] - self.vm_demand_ranges["ram"][0]) / (
                    self.vm_demand_ranges["ram"][1] - self.vm_demand_ranges["ram"][0]),
            'storage_norm_ratio': (dem["storage"] - self.vm_demand_ranges["storage"][0]) / (
                    self.vm_demand_ranges["storage"][1] - self.vm_demand_ranges["storage"][0]),
        }
        return features, graph_features

    def build_semantic_action_list(self):
        # order discrete actions by VMs with more demand first, and PMs with more capacity first
        vm_demands = []
        for vm_idx in range(self.n_vms):
            dem = self.alloc_graph.nodes[f"VM_{vm_idx}"]["x_dict"]["dem"]
            total_dem = sum(dem.values())
            vm_demands.append((vm_idx, total_dem))
        vm_demands.sort(key=lambda x: x[1], reverse=True)
        pm_capacities = []
        for pm_idx in range(self.n_pms):
            cap = self.alloc_graph.nodes[f"PM_{pm_idx}"]["x_dict"]["cap"]
            total_cap = sum(cap.values())
            pm_capacities.append((pm_idx, total_cap))
        pm_capacities.sort(key=lambda x: x[1], reverse=True)
        ordered_actions = []
        for vm_idx, _ in vm_demands:
            for pm_idx, _ in pm_capacities:
                ordered_actions.append((vm_idx, pm_idx))
        return ordered_actions

    def _get_obs(self):
        # Concatenate PM and VM features into a single observation vector
        pm_feats = []
        for idx in range(self.n_pms):
            pm_features, _ = self._pm_features(f"PM_{idx}")
            pm_feats.extend(pm_features.values())
        vm_feats = []
        for idx in range(self.n_vms):
            vm_features, _ = self._vm_features(f"VM_{idx}")
            vm_feats.extend(vm_features.values())
        return {"graph": np.array(pm_feats + vm_feats, dtype=np.float32)}

    def reset(self, seed=None, options=None):
        if self.revisiting:
            return self.reset_revisiting(seed=seed, options=options)
        else:
            return self.reset_non_revisiting(seed=seed, options=options)

    def step(self, vm_idx, pm_idx=None):
        if self.revisiting:
            return self.step_revisiting(vm_idx, pm_idx)
        else:
            return self.step_non_revisiting(vm_idx, pm_idx)

    def step_non_revisiting(self, vm_idx, pm_idx=None):

        action = super().pre_step((vm_idx, pm_idx))

        vms_changed = []
        pms_changed = []

        if not self.invalid_action and not self.no_action:

            vm_idx, pm_idx = action

            if isinstance(vm_idx, int):
                vm_idx = f"VM_{vm_idx}"

            if isinstance(pm_idx, int):
                pm_idx = f"PM_{pm_idx}"

            # --------------------------------------------------------
            # Same PM = invalid / useless movement
            # --------------------------------------------------------
            if pm_idx == self._get_pm_of_vm(vm_idx):

                self.same_pm_movements += 1

                self.reward = (
                        self.same_pm_penalty_sum /
                        min(
                            self.max_steps_overall,
                            self.max_steps_coefficient *
                            (self.n_vms + self.n_pms)
                        )
                )

            else:

                self.actual_movements += 1

                was_valid = self.is_valid_action(vm_idx, pm_idx)

                if was_valid:

                    self.actual_valid_movements += 1

                    movement_cost, vms_changed, pms_changed = (
                        self._place_vm(vm_idx, pm_idx)
                    )

                    # Once moved, this VM cannot be selected again
                    vm = int(vm_idx.split("_")[1])

                    self.action_list = [
                        a for a in self.action_list
                        if a[0] != vm
                    ]

                    self.semantic_action_list = [
                        a for a in self.semantic_action_list
                        if a[0] != vm
                    ]

                    self.alloc_graph.nodes[vm_idx]["x_dict"]["visited"] = True

                    # Update metrics BEFORE calculating reward
                    self.previous_metrics = self.current_metrics.copy()
                    self.current_metrics = self._compute_metrics()

                    self.reward = self._reward()

                    if self.add_movement_cost:
                        normalized_cost = min_max_normalize(
                            movement_cost,
                            0,
                            self.max_latency_movement_cost
                        )

                        self.reward -= (
                                self.movement_coefficient * normalized_cost
                        )

                        self.sum_movement_costs += normalized_cost

                else:

                    self.reward = (
                            self.invalid_action_penalty_sum /
                            min(
                                self.max_steps_overall,
                                self.max_steps_coefficient *
                                (self.n_vms + self.n_pms)
                            )
                    )

        # ------------------------------------------------------------
        # Termination: constructive mode
        # ------------------------------------------------------------

        mask = self.action_masks()

        self.done = (
                len(self.action_list) == 0
                or not any(mask)
        )

        if self.terminate_at_approximate_best:
            self.done |= (
                    self.current_metrics["weighted_sum"] >= self.best_score
            )

        # Update graph features
        self._update_changed_features(vms_changed, pms_changed)

        self.obs = self._get_obs()
        self.info = {}
        self.action = (vm_idx, pm_idx)

        super().post_step()
        self._record_metrics_history()

        return (
            self.obs,
            self.reward,
            self.done,
            self.truncated,
            self.info,
        )

    def _update_changed_features(self, vms_changed, pms_changed):

        for vm_id in vms_changed:
            _, vm_feature = self._vm_features(vm_id)

            for key in vm_feature:
                self.alloc_graph.nodes[vm_id]["x_dict"][key] = vm_feature[key]
                self.traffic_graph.nodes[vm_id]["x_dict"][key] = vm_feature[key]

        for pm_id in pms_changed:
            _, pm_feature = self._pm_features(pm_id)

            for key in pm_feature:
                self.alloc_graph.nodes[pm_id]["x_dict"][key] = pm_feature[key]
                self.latency_graph.nodes[pm_id]["x_dict"][key] = pm_feature[key]

    def step_revisiting(self, vm_idx, pm_idx=None):

        action = super().pre_step((vm_idx, pm_idx))

        vms_changed = []
        pms_changed = []

        if not self.invalid_action and not self.no_action:

            vm_idx, pm_idx = action

            if isinstance(vm_idx, int):
                vm_idx = f"VM_{vm_idx}"

            if isinstance(pm_idx, int):
                pm_idx = f"PM_{pm_idx}"

            # --------------------------------------------------------
            # Same PM
            # --------------------------------------------------------

            if pm_idx == self._get_pm_of_vm(vm_idx):

                self.same_pm_movements += 1

                self.reward = (
                        self.same_pm_penalty_sum /
                        min(
                            self.max_steps_overall,
                            self.max_steps_coefficient *
                            (self.n_vms + self.n_pms)
                        )
                )

            else:

                self.actual_movements += 1

                was_valid = self.is_valid_action(vm_idx, pm_idx)

                if was_valid:

                    self.actual_valid_movements += 1

                    movement_cost, vms_changed, pms_changed = (
                        self._place_vm(vm_idx, pm_idx)
                    )

                    # ------------------------------------------------
                    # IMPORTANT:
                    # Do NOT remove this VM from action_list.
                    # Revisiting allows the same VM to move again.
                    # ------------------------------------------------

                    self.alloc_graph.nodes[vm_idx]["x_dict"]["visited"] = True

                    # ------------------------------------------------
                    # Calculate new metrics
                    # ------------------------------------------------

                    self.previous_metrics = self.current_metrics.copy()
                    self.current_metrics = self._compute_metrics()

                    # ------------------------------------------------
                    # Local transition reward:
                    #
                    # reward = current score - previous score
                    # ------------------------------------------------

                    self.reward = self._reward()

                    if self.add_movement_cost:
                        normalized_cost = min_max_normalize(
                            movement_cost,
                            0,
                            self.max_latency_movement_cost
                        )

                        self.reward -= (
                                self.movement_coefficient * normalized_cost
                        )

                        self.sum_movement_costs += normalized_cost

                    # ------------------------------------------------
                    # Best-so-far tracking
                    # ------------------------------------------------

                    current_score = self.current_metrics["weighted_sum"]

                    if current_score > self.best_episode_score:
                        self.best_episode_score = current_score
                        self.best_episode_allocation = (
                            self._encode_allocation()
                        )

                else:

                    self.reward = (
                            self.invalid_action_penalty_sum /
                            min(
                                self.max_steps_overall,
                                self.max_steps_coefficient *
                                (self.n_vms + self.n_pms)
                            )
                    )

        # ------------------------------------------------------------
        # Revisiting termination:
        # fixed exploration horizon
        # ------------------------------------------------------------

        # Optional approximate-best termination
        if self.terminate_at_approximate_best:
            self.done |= (
                    self.current_metrics["weighted_sum"] >= self.best_score
            )

        # Update graph features
        self._update_changed_features(
            vms_changed,
            pms_changed
        )

        self.obs = self._get_obs()
        self.info = {}
        self.action = (vm_idx, pm_idx)

        super().post_step()
        self._record_metrics_history()

        return (
            self.obs,
            self.reward,
            self.done,
            self.truncated,
            self.info,
        )

    def reset_non_revisiting(self, seed=None, options=None):
        super().pre_reset()

        # Start from the same feasible empirical allocation
        # used to initialize the environment.
        self.alloc_graph = copy.deepcopy(self.initial_alloc_graph)

        self.sum_movement_costs = 0.0
        self.same_pm_movements = 0
        self.actual_movements = 0
        self.actual_valid_movements = 0

        # Reset visited flags
        for vm in range(self.n_vms):
            self.alloc_graph.nodes[f"VM_{vm}"]["x_dict"]["visited"] = False

        # Compute initial metrics
        self._initial_metrics = self._compute_metrics()
        self.initial_score = self.compute_weighted_sum_normalized_metrics(
            self._initial_metrics
        )

        self.previous_metrics = self._initial_metrics.copy()
        self.current_metrics = self._initial_metrics.copy()
        self.metrics_history = []
        self.reward = 0.0
        self._record_metrics_history()

        # Every VM is initially available to be moved once
        self.action_list = self.build_action_list()
        self.semantic_action_list = self.build_semantic_action_list()

        self.steps = 0

        # Observation
        self.obs = self._get_obs()
        self.info = {}

        super().post_reset()

        return self.obs, self.info

    def _random_allocation(self, max_attempts=100):
        """
        Generate a random feasible VM -> PM allocation.

        Retries the complete random construction if a greedy random
        placement gets stuck.
        """

        for _ in range(max_attempts):

            # Reset PM usage
            for pm_idx in range(self.n_pms):
                pm = f"PM_{pm_idx}"

                self.alloc_graph.nodes[pm]["x_dict"]["used"] = {
                    k: 0 for k in self.pm_capacity_ranges
                }

                self.alloc_graph.nodes[pm]["x_dict"]["power"] = self.p_idle

            # Remove current VM-PM allocations
            self.alloc_graph.remove_edges_from(
                list(self.alloc_graph.edges())
            )

            # Random VM ordering
            vm_indices = list(range(self.n_vms))
            random.shuffle(vm_indices)

            success = True

            for vm_idx in vm_indices:

                vm = f"VM_{vm_idx}"

                feasible_pms = [
                    f"PM_{pm_idx}"
                    for pm_idx in range(self.n_pms)
                    if self.is_valid_action(vm, pm_idx)
                ]

                if not feasible_pms:
                    success = False
                    break

                pm = random.choice(feasible_pms)

                self._place_vm(vm, pm)

            if success:
                return self._encode_allocation()

        raise RuntimeError(
            f"Failed to generate a random feasible allocation "
            f"after {max_attempts} attempts."
        )

    def reset_revisiting(self, seed=None, options=None):
        super().pre_reset()

        # ------------------------------------------------------------
        # Random initial feasible allocation
        # ------------------------------------------------------------
        self.alloc_graph = copy.deepcopy(self.initial_alloc_graph)

        # ------------------------------------------------------------
        # Reset bookkeeping
        # ------------------------------------------------------------
        self.sum_movement_costs = 0.0
        self.same_pm_movements = 0
        self.actual_movements = 0
        self.actual_valid_movements = 0

        for vm_idx in range(self.n_vms):
            self.alloc_graph.nodes[
                f"VM_{vm_idx}"
            ]["x_dict"]["visited"] = False

        # ------------------------------------------------------------
        # Initial metrics
        # ------------------------------------------------------------

        self._initial_metrics = self._compute_metrics()

        self.initial_score = (
            self.compute_weighted_sum_normalized_metrics(
                self._initial_metrics
            )
        )

        self.previous_metrics = self._initial_metrics.copy()
        self.current_metrics = self._initial_metrics.copy()
        self.metrics_history = []
        self.reward = 0.0
        self._record_metrics_history()

        # ------------------------------------------------------------
        # Best solution at beginning of episode
        # ------------------------------------------------------------

        self.best_episode_score = self.initial_score
        self.best_episode_allocation = self._encode_allocation()

        # ------------------------------------------------------------
        # Revisiting: all VM-PM actions remain available
        # ------------------------------------------------------------

        self.action_list = self.build_action_list()
        self.semantic_action_list = self.build_semantic_action_list()

        self.obs = self._get_obs()
        self.info = {}

        super().post_reset()

        return self.obs, self.info

    def save_step_log(self):
        super().save_step_log()
        self.step_logs[-1].update({
            'observation': self.obs,
            'alloc_graph_nodes': graph_nodes_to_text(self.alloc_graph),
            'alloc_graph_edges': graph_edges_to_text(self.alloc_graph),
            'action': str(self.action),
        })
        for metric in self.current_metrics:
            self.step_logs[-1][metric] = self.current_metrics[metric]

    def _compute_metrics(self, compute_norm=True):
        # Compute raw and normalized environment metrics
        util_scores = []
        power_scores = []
        occupancies = {}
        for i in range(self.n_pms):
            pm_name = f"PM_{i}"
            info = self.alloc_graph.nodes[pm_name]["x_dict"]

            total_cap = sum(info["cap"].values())
            balance = sum(
                min(info["used"][k], info["cap"][k] - info["used"][k])
                for k in info["cap"]
            )

            util_scores.append(1 - (balance / total_cap * 2))
            power_scores.append(self._compute_power(pm_name))

            for k in info["cap"]:
                occupancies.setdefault(k, []).append(info["used"][k] / info["cap"][k])

        metrics = {
            "avg_util": np.mean(util_scores),
            "avg_power": np.mean(power_scores),
        }

        # --------------------
        # Security metric
        # --------------------
        security_risks = []
        for i in range(self.n_pms):
            pm_name = f"PM_{i}"
            pm_info = self.alloc_graph.nodes[pm_name]["x_dict"]

            colocated_vms = [
                nbr for nbr in self.alloc_graph.neighbors(pm_name)
                if self.alloc_graph.nodes[nbr]["x_dict"]["type"] == "VM"
            ]

            pm_risk = 0.0
            for vm in colocated_vms:
                vm_info = self.alloc_graph.nodes[vm]["x_dict"]
                pm_risk += (
                        vm_info["vuln_prob"]
                        * pm_info["escape_prob"]
                        * (len(colocated_vms) - 1)
                )

            security_risks.append(pm_risk)

        metrics["avg_security_risk"] = np.mean(security_risks)

        # --------------------
        # Communication load
        # --------------------
        tenant_latencies = {}
        for vm in self.traffic_graph.nodes:
            tenant = self.traffic_graph.nodes[vm]["x_dict"]["tenant"]
            tenant_latencies.setdefault(tenant, [])

            for nbr in self.traffic_graph.neighbors(vm):
                pm1 = self._get_pm_of_vm(vm)
                pm2 = self._get_pm_of_vm(nbr)

                traffic = self._edge_value(self.traffic_graph[vm][nbr], "traffic", 0.0)
                if pm1 != pm2:
                    lat = self.latency_dict[pm1][pm2]
                    tenant_latencies[tenant].append(lat * traffic)

        avg_load = np.mean([
            np.mean(v) if v else 0.0
            for v in tenant_latencies.values()
        ]) if tenant_latencies else 0.0

        metrics["avg_load"] = avg_load

        # --------------------
        # Packing efficiency
        # --------------------
        vms_per_pm = [
            sum(
                1 for nbr in self.alloc_graph.neighbors(f"PM_{i}")
                if self.alloc_graph.nodes[nbr]["x_dict"]["type"] == "VM"
            )
            for i in range(self.n_pms)
        ]

        active_pms = sum(1 for x in vms_per_pm if x > 0)
        vm_density = self.n_vms / active_pms if active_pms > 0 else 0
        baseline_density = self.n_vms / self.n_pms

        packing_efficiency = (
            vm_density / baseline_density if baseline_density > 0 else 0
        )

        metrics["packing_efficiency"] = packing_efficiency
        metrics["vm_density"] = vm_density
        metrics["active_pms"] = active_pms
        # compute normalized version of each
        if compute_norm and hasattr(self, 'metric_bounds'):
            for m in ["avg_util",
                      "packing_efficiency",
                      "avg_power",
                      "avg_load",
                      "avg_security_risk"]:
                v = metrics[m]
                min_v, max_v = self.metric_bounds[m]

                # Min-max normalization
                norm = min_max_normalize(v, min_v, max_v)
                # Invert if lower-is-better
                if m in self.minimize_metrics:
                    norm = 1.0 - norm

                metrics[f"norm_{m}"] = norm
            metrics['weighted_sum'] = self.compute_weighted_sum_normalized_metrics(metrics)
        return metrics

    def _reward(self):
        m = self.current_metrics
        pm = self.previous_metrics

        current_score = m["weighted_sum"]
        previous_score = pm["weighted_sum"]
        # make current - worst / best - worst
        reward = (current_score - previous_score) / (self.best_score - self.initial_score + 1e-8)
        return reward

    def update_metrics(self):
        self._metrics = {}
        for k in self.current_metrics:
            self._metrics[k + "_difference"] = self.current_metrics[k] - self._initial_metrics[k]
        self._metrics['avg_norm_movement_cost'] = self.sum_movement_costs / max(1,
                                                                                self.current_step - self.count_no_actions) if self.current_step - self.count_no_actions > 0 else 0.0
        self._metrics['actual_movements'] = self.actual_movements / self.current_step if self.current_step > 0 else 0.0
        self._metrics[
            'actual_valid_movements'] = self.actual_valid_movements / self.current_step if self.current_step > 0 else 0.0
        self._metrics[
            'same_pm_movements'] = self.same_pm_movements / self.current_step if self.current_step > 0 else 0.0


class ExtendedVMPlacementEnv(VMPlacementEnv, ContinuousEnv):
    def __init__(self, **config):
        super().__init__(**config)
        self.update_spec()

    def update_spec(self):
        self._observation_type = {
            'alloc_graph': Graph(poolings=[MeanPooling, SumPooling, MinPooling, MaxPooling], graph_name='alloc_graph'),
            'alloc_nodes_number': Function(func=get_number_nodes, graph_name='alloc_graph'),
            'alloc_edges_number': Function(func=get_number_edges, graph_name='alloc_graph'),
            'alloc_average_node_degree': Function(func=get_average_node_degree, graph_name='alloc_graph'),
            'alloc_graph_density': Function(func=get_graph_density, graph_name='alloc_graph'),
            'traffic_graph': Graph(poolings=[MeanPooling, SumPooling, MinPooling, MaxPooling],
                                   graph_name='traffic_graph'),
            'traffic_nodes_number': Function(func=get_number_nodes, graph_name='traffic_graph'),
            'traffic_edges_number': Function(func=get_number_edges, graph_name='traffic_graph'),
            'traffic_average_node_degree': Function(func=get_average_node_degree, graph_name='traffic_graph'),
            'traffic_graph_density': Function(func=get_graph_density, graph_name='traffic_graph'),
            'latency_graph': Graph(poolings=[MeanPooling, SumPooling, MinPooling, MaxPooling],
                                   graph_name='latency_graph'),
            'latency_nodes_number': Function(func=get_number_nodes, graph_name='latency_graph'),
            'latency_edges_number': Function(func=get_number_edges, graph_name='latency_graph'),
            'latency_average_node_degree': Function(func=get_average_node_degree, graph_name='latency_graph'),
            'latency_graph_density': Function(func=get_graph_density, graph_name='latency_graph'),
            'action_space': ActionSpace(poolings=[MeanPooling, SumPooling, MaxPooling, MinPooling])
        }
        if self.revisiting:
            self._action_type = {
                'VM': Node(spec={"type": "VM"}, graph_name='alloc_graph'),
                'PM': Node(spec={"type": "PM"}, graph_name='alloc_graph')
            }
        else:
            self._action_type = {
                'VM': Node(spec={"type": "VM", "visited": False}, graph_name='alloc_graph'),
                'PM': Node(spec={"type": "PM"}, graph_name='alloc_graph')
            }
        if self.no_action_support:
            self.discrete_actions = ["no_action"]
        else:
            self.discrete_actions = []
        self._node_attributes = {
            'type': Attribute(feature_extractor=OneHotEncoding,
                              feature_extractor_args={'set': ['PM', 'VM']},
                              encoding=BinaryValue(),
                              embedding_size=2,
                              graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),
            'visited': Attribute(
                encoding=BinaryValue(),
                embedding_size=1,
                graph_name=['alloc_graph']
            ),
            'tenant': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=0,
                max_value=self.n_tenants - 1
            ),
                embedding_size=1,
                graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),
            'pe_norm_ratio': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=0.0,
                max_value=1.0
            ),
                embedding_size=1,
                graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),
            'mips_norm_ratio': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=0.0,
                max_value=1.0
            ),
                embedding_size=1,
                graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),
            'ram_norm_ratio': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=0.0,
                max_value=1.0
            ),
                embedding_size=1,
                graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),
            'storage_norm_ratio': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=0.0,
                max_value=1.0
            ),
                embedding_size=1,
                graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),
            'pe_ratio': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=0.0,
                max_value=1.0
            ),
                embedding_size=1,
                graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),
            'mips_ratio': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=0.0,
                max_value=1.0
            ),
                embedding_size=1,
                graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),
            'ram_ratio': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=0.0,
                max_value=1.0
            ),
                embedding_size=1,
                graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),
            'storage_ratio': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=0.0,
                max_value=1.0
            ),
                embedding_size=1,
                graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),
            'power_ratio': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=0.0,
                max_value=1.0
            ),
                embedding_size=1,
                graph_name=['alloc_graph', 'traffic_graph', 'latency_graph']),

        }
        self._edge_attributes = {
            'traffic': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=self.min_traffic,
                max_value=self.max_traffic
            ),
                embedding_size=1,
                graph_name='traffic_graph'),
            'latency': Attribute(encoding=ContinuousValue(
                normalization="min_max",
                min_value=self.latency_min,
                max_value=self.latency_max
            ),
                embedding_size=1,
                graph_name='latency_graph'),
        }
        self.action_candidates_reconstruction = False
        self.reconstruct_only_changed_nodes = False
        self.action_space_reconstruction_each_step = True

    def sample_valid_action(self):
        """Sample a VM to migrate and a feasible PM."""
        vm = random.randrange(self.n_vms)
        fits = []
        for pm in range(self.n_pms):
            if self.is_valid_action(f"VM_{vm}", f"PM_{pm}"):
                fits.append(pm)
        if not fits:
            return None
        pm = random.choice(fits)
        return f"VM_{vm}", f"PM_{pm}"

    def get_graphs(self):
        return {
            'alloc_graph': self.alloc_graph,
            'traffic_graph': self.traffic_graph,
            'latency_graph': self.latency_graph
        }