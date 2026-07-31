#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

from utils.features_utils import get_number_nodes, get_number_edges, get_average_node_degree, get_graph_density
from utils.model import Node, Graph, Object, Attribute, Function, ActionSpace
from utils.feature_extractors import OneHotEncoding, LanguageModelEmbedding
from utils.pooling_functions import MeanPooling, SumPooling, MinPooling, MaxPooling
from utils.encoding_options import ContinuousValue, BinaryValue, MultiClassValue
from wrappers.wrapper import ContinuousEnv
from gymnasium.spaces import Box, Dict
import numpy as np
import networkx as nx
import copy
from collections import defaultdict
from utils.log_utils import graph_nodes_to_text, graph_edges_to_text
from utils.cyberattack_utils import sample_service_vuln_pairs, assign_pairs_to_nodes
import random
from envs.env import DiscreteEnv

class CyberAttackEnv(DiscreteEnv):
    """
        Cyberattack Environment.

        Agent selects source node, target node, vulnerability to exploit (and optionally outcome) to perform an attack action. The environment checks the validity of the action based on the true graph configuration and updates the state of the environment accordingly, including the attack graph visible to the agent, compromised nodes, and other features. Rewards are given based on the outcomes of the action and whether it was valid or not. The episode can terminate based on certain conditions such as achieving control over all nodes.
        The environment is highly configurable with parameters controlling the graph structure, vulnerability allocation, reward models, and more, allowing for a wide range of scenarios to be simulated.
    """
    def __init__(
        self,
        n_nodes,
        n_vulns_per_node,
        vulns_overlap, # percentage of vulns that should be shared between nodes, controls clustering
        huggingface, # key for hugging face to load LM
        services_dict, # list of service-vulnerability pairs to sample from
        reward_models, # dict of threat models and reward mapping associated
        goal, # used to select reward model from reward_models dict
        p_communication_edge=0.3,
        max_dos_percentage=0.25,
        max_lateral_percentage=0.25,
        p_data_present=0.5,
        p_feature_visible=0.2,
        p_patch_scale_year=0.3,
        p_recon=0.7,
        p_detection=0.2,
        fully_connected_graph=True, # communication graph
        invalid_action_penalty_sum=-10,
        outcome_selection=False, # whether to include outcome in action space and reward based on it, or just assume all outcomes happen and reward based on that
        outcome_selection_all=False, # all outcomes appear in the action space, but many are invalid
        remove_DOS_starter=False, # remove DOS vulns of starter node -> they can block episode as agent will have 0 nodes controlled
        remove_all_DOS=False, # remove DOS vulns from all nodes -> to test agent performance without disruption of nodes
        **kwargs
    ):
        super().pre_init(**kwargs)
        self.n_nodes = n_nodes
        self.n_vulns_per_node = n_vulns_per_node
        self.vulns_overlap = vulns_overlap
        self.huggingface = huggingface
        self.services_dict = services_dict
        self.p_communication_edge = p_communication_edge
        self.max_dos_percentage = max_dos_percentage
        self.max_lateral_percentage = max_lateral_percentage
        self.p_data_present = p_data_present
        self.p_feature_visible = p_feature_visible
        self.p_patch_scale_year = p_patch_scale_year
        self.p_recon = p_recon
        self.p_detection = p_detection
        self.fully_connected_graph = fully_connected_graph
        self.invalid_action_penalty_sum = invalid_action_penalty_sum
        self.outcome_selection = outcome_selection
        self.outcome_selection_all = outcome_selection_all
        self.remove_DOS_starter = remove_DOS_starter
        self.remove_all_DOS = remove_all_DOS

        self.reward_map = reward_models.get(goal, {})
        self.outcomes = [
            "lateral move", "reconnaissance",
            "discovery", "collection", "exfiltration", "DOS",
            "persistence", "defense evasion", "privilege escalation",
        ]
        self.node_feature_dim = 11 + len(self.outcomes)
        self.scenario_size = n_nodes

        # Vulnerability allocation
        self.true_graph = self._generate_true_graph()
        self.initial_true_graph = copy.deepcopy(self.true_graph)
        # Observations: flattened node features
        self.observation_space = Dict({
            "graph": Box(
                low=0,
                high=100,
                shape=(self.n_nodes * self.node_feature_dim,),
                dtype=np.float32
            )
        })

        # Action space: (src, tgt, cve_idx, outcome_idx)
        self.all_cves = self._flatten_cves(self.services_dict)
        # mantain mapping description to outcomes and ID (description is action in projection)
        self.outcomes_per_cve = {
            cve['description']: cve['outcomes'] for cve in self.all_cves
        }
        self.cve_desc_to_id = defaultdict(list)
        for cve in self.all_cves:
            if not cve['cve_id'] in self.cve_desc_to_id[cve['description']]:
                self.cve_desc_to_id[cve['description']].append(cve['cve_id'])
        # action space construction
        self.semantic_action_list = self.build_semantic_action_list()
        self.action_list = self.build_action_list()
        self.reset()
        self.post_init()

    def build_action_list(self):
        # build list of all valid actions based on current true graph configuration, to be used for action masking and semantic ordering
        actions = []
        for src in range(self.n_nodes):
            for tgt in range(self.n_nodes):
                tgt_vulns = self.true_graph.nodes[tgt]['x_dict']['vulns']
                for vuln in tgt_vulns:
                    cve_id = vuln['cve_id']
                    outcomes = vuln['outcomes']  # outcome list for this vuln ONLY
                    if self.outcome_selection_all:
                        for outcome in self.outcomes:
                            actions.append((src, tgt, cve_id, outcome))
                    elif self.outcome_selection:
                        for outcome in outcomes:
                            actions.append((src, tgt, cve_id, outcome))
                    else:
                        actions.append((src, tgt, cve_id))
        return actions

    def update_config(self, config):
        super().update_config(config)
        self.goal = config['goal']
        self.huggingface = config['huggingface']
        self.outcome_selection = config.get('outcome_selection', self.outcome_selection)
        self.outcome_selection_all = config.get('outcome_selection_all', self.outcome_selection_all)
        self.remove_DOS_starter = config.get('remove_DOS_starter', self.remove_DOS_starter)
        self.remove_all_DOS = config.get('remove_all_DOS', self.remove_all_DOS)
        # will affect action space so need to rebuild
        self.action_list = self.build_action_list()
        self.semantic_action_list = self.build_semantic_action_list()

    def save_step_log(self):
        super().save_step_log()
        if not hasattr(self, 'step_logs'):
            self.step_logs = []
        self.step_logs[-1].update({
            'observation': self.obs,
            'attack_graph_nodes':  graph_nodes_to_text(self.attack_graph),
            'attack_graph_edges': graph_edges_to_text(self.attack_graph),
            'action': str(self.action),
            'owned_nodes': list(self.compromised_nodes),
            'dos_nodes': list(self.dos_nodes),
            'visible_nodes': list(self.visible_nodes),
            'outcomes_executed': self.outcomes_executed
        })

    def build_semantic_action_list(self):
        # create an action space ordered properly based on some metrics
        mapping = []
        # order actions smartly, first source-target with more vulns
        node_vuln_counts = {
            node: len(self.true_graph.nodes[node]['x_dict']['vulns'])
            for node in self.true_graph.nodes()
        }
        sorted_nodes = sorted(node_vuln_counts.items(), key=lambda x: x[1], reverse=True)
        for src, _ in sorted_nodes:
            for tgt, _ in sorted_nodes:
                tgt_vulns = self.true_graph.nodes[tgt]['x_dict']['vulns']
                for vuln in tgt_vulns:
                    cve_id = vuln['cve_id']
                    outcomes = vuln['outcomes']
                    if self.outcome_selection_all:
                        for outcome in self.outcomes:
                            mapping.append((src, tgt, cve_id, outcome))
                    elif self.outcome_selection:
                        for outcome in outcomes:
                            mapping.append((src, tgt, cve_id, outcome))
                    else:
                        mapping.append((src, tgt, cve_id))
        return mapping


    def _flatten_cves(self, services_dict):
        # services_dict is a list of SERVICE–VULN PAIRS
        cve_list = []
        seen = set()
        for pair in services_dict:
            services = pair.get("services", [])
            vulns = pair.get("vulns", [])

            assert services, "Pair has no services"
            assert vulns, "Pair has no vulnerabilities"

            for v in vulns:
                cve_id = v["cve_id"]
                # Prevent duplicates if pairs are reused
                if cve_id in seen:
                    continue
                seen.add(cve_id)

                # Associate CVE with all services in the pair
                service_names = [
                    f"{s['product']}_{s['version']}" for s in services
                ]

                cve_list.append({
                    "service": service_names,  # NOTE: now a LIST, not a single string
                    "cve_id": cve_id,
                    "description": v.get("description", cve_id),
                    "outcomes": [o for o in v.get("outcomes", []) if o != "execution"],
                    "metrics": v.get("metrics", {}),
                    "priv_required": any(
                        m.get("obtainAllPrivilege", False)
                        or m.get("obtainUserPrivilege", False)
                        for m in v.get("metrics", {}).get("cvssMetricV2", [])
                    )
                })
        return cve_list

    def reset(self, seed=None, options=None):
        # reset function to initialize the environment state, including generating a new true graph based on the parameters and setting the initial compromised node and visible nodes. Also resets all metrics and counters.
        super().pre_reset()
        self.compromised_nodes = set()
        self.dos_nodes = set()
        self.visible_nodes = set()
        self.true_graph = copy.deepcopy(self.initial_true_graph)
        self.attack_graph = nx.Graph()
        self.privilege_escalation_invalid_actions = 0
        self.valid_actions_per_outcome = {outcome: 0 for outcome in self.outcomes}
        self.repeated_actions_per_outcome = {outcome: 0 for outcome in self.outcomes}
        self.privileges_invalid_actions = 0
        self.blocked_actions_defense = 0
        self.attack_graph_edges = 0

        # Pick random start node
        start_node = random.randint(0, self.n_nodes - 1)
        # remove all DOS actions from starter node
        if self.remove_DOS_starter:
            for v in self.true_graph.nodes[start_node]['x_dict']['vulns']:
                if 'DOS' in v['outcomes']:
                    v['outcomes'] = [o for o in v['outcomes'] if o != 'DOS']

        # remove all DOS actions from all nodes if specified, to test agent performance without disruption of nodes (can be set to True for starter node and False for rest to just remove DOS from starter)
        if self.remove_all_DOS:
            for node in self.true_graph.nodes():
                for v in self.true_graph.nodes[node]['x_dict']['vulns']:
                    if 'DOS' in v['outcomes']:
                        v['outcomes'] = [o for o in v['outcomes'] if o != 'DOS']

        # initialize features of start node to reflect initial compromise and visibility
        node_dict = self.true_graph.nodes[start_node]['x_dict']
        node_dict['compromised'] = True
        node_dict['privilege_level'] = 'user'
        node_dict['graph_visible'] = True

        self.visible_nodes.add(start_node)
        self.compromised_nodes.add(start_node)
        self.attack_graph.add_node(start_node)

        # provide initial set of possible target nodes
        self._reveal_neighbors(start_node, self.p_recon)
        # check if no node has been revealed, if so reveal one random neighbor
        if len(self.visible_nodes) == 1:
            neighbors = list(self.true_graph.neighbors(start_node))
            if neighbors:
                forced_node = random.choice(neighbors)
                n_dict = self.true_graph.nodes[forced_node]['x_dict']
                n_dict['graph_visible'] = True
                self.visible_nodes.add(forced_node)
                self.attack_graph.add_node(forced_node)
        if len(self.visible_nodes) == 1:
            all_nodes = set(self.true_graph.nodes())
            all_nodes.remove(start_node)
            forced_node = random.choice(list(all_nodes))
            n_dict = self.true_graph.nodes[forced_node]['x_dict']
            n_dict['graph_visible'] = True
            self.visible_nodes.add(forced_node)
            self.attack_graph.add_node(forced_node)

        # initialize attack graph visible to the agent based on true graph
        self.copy_true_into_attack_graph()

        # self-loop for initial access, to have one edge at start
        self.attack_graph.add_edge(start_node, start_node)
        self.attack_graph.edges[start_node, start_node]['edge_attr_dict'] = {}
        self.attack_graph.edges[start_node, start_node]['edge_attr_dict']['exploited_vulns'] = ['initial_access']

        self.obs = self._get_obs()
        self.info = {}
        super().post_reset()
        return self.obs, {}

    def _generate_true_graph(self):
        # generate graph of the scenario instance based on the parameters
        if self.fully_connected_graph:
            G = nx.complete_graph(self.n_nodes)
        else:
            while True:
                G = nx.erdos_renyi_graph(self.n_nodes, self.p_communication_edge)
                if nx.is_connected(G):
                    break

        # sample possible vuln configurations (composed of service tuples) to be allocated with a certain overlap
        service_vuln_pairs = sample_service_vuln_pairs(
                self.services_dict,
                self.n_vulns_per_node,
                max_dos_percentage=self.max_dos_percentage,
                max_lateral_percentage=self.max_lateral_percentage
            )

        if not service_vuln_pairs:
            raise ValueError("Unable to sample service-vulnerability pairs with the given constraints.")

        # respect overlap
        node_pairs = assign_pairs_to_nodes(
            service_vuln_pairs,
            self.n_nodes,
            self.vulns_overlap
        )

        # Ensure some features based on the allocated vulnerabilities
        for node, pair in zip(G.nodes(), node_pairs):
            data_discoverable = False
            info_discoverable = False
            services = pair["services"]
            vulns = pair["vulns"]

            # Ensure credential access
            if not any("credential access" in v["outcomes"] for v in vulns):
                candidates = [
                    v for p in service_vuln_pairs
                    for v in p["vulns"]
                    if "credential access" in v["outcomes"]
                ]
                if candidates:
                    vulns[-1] = random.choice(candidates)

            if "discovery" in [o for v in vulns for o in v["outcomes"]]:
                info_discoverable = True
            if "collection" in [o for v in vulns for o in v["outcomes"]]:
                data_discoverable = True
            G.nodes[node]["x_dict"] = {
                "services": services,
                "vulns": vulns,
                "graph_visible": False,
                "feature_visible": info_discoverable and random.random() < self.p_feature_visible,
                "compromised": False,
                "privilege_level": "none",
                "data_present": data_discoverable and random.random() < self.p_data_present,
                "data_collected": False,
                "data_exfiltrated": False,
                "persistent": False,
                "dos": False,
                "defense_evaded": False,
            }

            assert len(vulns) == self.n_vulns_per_node

        self.services_dict = service_vuln_pairs
        return G

    def _reveal_neighbors(self, node, p_recon=1.0):
        # reveal neighbors of the node with probability p_recon, return number of nodes revealed
        num_nodes_revealed = 0
        for neighbor in self.true_graph.neighbors(node):
            n_dict = self.true_graph.nodes[neighbor]['x_dict']

            if not n_dict['graph_visible'] and random.random() < p_recon:
                n_dict['graph_visible'] = True
                self.visible_nodes.add(neighbor)
                self.attack_graph.add_node(neighbor)
                num_nodes_revealed += 1
        self.copy_true_into_attack_graph()

        return num_nodes_revealed

    def copy_true_into_attack_graph(self):
        # paste everything from true graph into attack graph
        for node in self.attack_graph.nodes:
            self.attack_graph.nodes[node]['x_dict'] = copy.deepcopy(self.true_graph.nodes[node]['x_dict'])

    def _get_obs(self):
        # for each node, return feature vector based on its true graph attributes and whether it's visible or not (if not visible, most features are masked)
        features = []
        for i in range(self.n_nodes):
            if i in self.true_graph.nodes:
                n = self.true_graph.nodes[i]['x_dict']
                # Count outcomes and privilege-required
                total_possible_outcomes = {}
                priv_required_count = 0
                for v in n['vulns']:
                    for o in v['outcomes']:
                        total_possible_outcomes[o] = total_possible_outcomes.get(o, 0) + 1
                    priv_required_count += v.get('priv_required', 0)
                if n['dos']:
                    vec = [0 for _ in range(10 + len(self.outcomes) + 1)]
                elif n['feature_visible'] and n['graph_visible']:
                    vec = [
                        1.0, # feature visible
                        1.0 if n['compromised'] else 0.0,
                        len(n['services']),
                        len(n['vulns']),
                        1.0 if n['privilege_level']=='root' else 0.0,
                        1.0 if n['data_present'] else 0.0,
                        1.0 if n['data_collected'] else 0.0,
                        1.0 if n['data_exfiltrated'] else 0.0,
                        1.0 if n['persistent'] else 0.0,
                        1.0 if n['defense_evaded'] else 0.0,
                        priv_required_count
                    ]
                    for outcome in self.outcomes:
                        vec.append(total_possible_outcomes.get(outcome, 0))
                elif n['graph_visible']:
                    vec = [
                        0.0, # feature visible
                        1.0 if n['compromised'] else 0.0,
                        0.0, 0.0,
                        0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0,
                        0.0
                    ]
                    for _ in self.outcomes:
                        vec.append(0)
                else:
                    vec = [0.0 for _ in range(10 + len(self.outcomes) + 1)]
            else:
                raise ValueError("Node index exceeds true graph nodes.")
            features.extend(vec)
        return {"graph": np.array(features, dtype=np.float32)}

    def is_valid_action(self, src, tgt, cve_id, outcome=None):
        # action validity checks based on true graph configuration, return True if valid, False if not.
        # Also set self.cve_info and self.cve_outcomes for reward calculation in step function to avoid redundant lookups.
        self.cve_info = None
        self.cve_outcomes = None
        if src not in self.true_graph.nodes or tgt not in self.true_graph.nodes:
            if self.verbose > 2:
                print("Source or target node not in true graph.")
            return False

        if src not in self.attack_graph.nodes or tgt not in self.attack_graph.nodes:
            if self.verbose > 2:
                print("Source or target node not in attack graph.")
            return False

        src_info = self.true_graph.nodes[src]["x_dict"]
        tgt_info = self.true_graph.nodes[tgt]["x_dict"]

        # DOS checks
        if src_info["dos"] or tgt_info["dos"]:
            if self.verbose > 2:
                print("Source or target node is in DOS state.")
            return False

        # visibility checks
        if (not src_info["graph_visible"]) or (not tgt_info["graph_visible"]):
            if self.verbose > 2:
                print("Source or target node is not visible.")
            return False

        # source must be compromised
        if not src_info["compromised"]:
            if self.verbose > 2:
                print("Source node is not compromised.")
            return False

        # CVE must exist on target
        cve_info = None
        for vuln in tgt_info["vulns"]:
            if vuln["cve_id"] == cve_id:
                cve_info = vuln
                break

        if cve_info is None:
            if self.verbose > 2:
                print("CVE ID not found in target node vulnerabilities.")
            return False
        self.cve_info = cve_info
        self.cve_outcomes = cve_info["outcomes"]

        # outcome checks
        if self.outcome_selection or self.outcome_selection_all:
            if outcome is None:
                if self.verbose > 2:
                    print("Outcome is required but None was provided.")
                return False

            if outcome not in cve_info["outcomes"]:
                if self.verbose > 2:
                    print("Outcome not valid for the selected CVE.")
                return False

            return True

        return True

    def step(self, src, tgt=None, cve_id=None, outcome=None):
        decoded_action = super().pre_step(src, tgt, cve_id, outcome)

        self.outcomes_executed = []
        if not self.no_action and not self.invalid_action:
            if self.outcome_selection or self.outcome_selection_all:
                src, tgt, cve_id, outcome = decoded_action
            else:
                src, tgt, cve_id = decoded_action

            if "CVE-" in cve_id:
                pass
            else:
                # continuous case where cve_id is description
                cve_ids = self.cve_desc_to_id[cve_id]
                found = False
                right_cve_id = None
                for cid in cve_ids:
                    for v in self.true_graph.nodes[tgt]['x_dict']['vulns']:
                        if v['cve_id'] == cid:
                            right_cve_id = cid
                            found = True
                            break
                    if found:
                        break
                cve_id = right_cve_id

            # pick cve_info from target node
            tgt_dict = self.true_graph.nodes[tgt]['x_dict']
            valid = self.is_valid_action(src, tgt, cve_id, outcome)
            if valid:
                # Check CVE exists in node
                blocked = False
                reward = 0
                if self.cve_info.get('priv_required', False) and tgt_dict['privilege_level'] != 'root':
                    reward = self.reward_map.get('no_enough_privileges', 0)
                    blocked = True
                    self.privileges_invalid_actions += 1
                # elif there is defense not evaded make it probabilistically fail
                if not tgt_dict['defense_evaded']:
                    if random.random() < self.p_detection:
                        reward = self.reward_map.get('detected_and_blocked', 0)
                        blocked = True
                        self.blocked_actions_defense += 1
                        if self.verbose > 2:
                            print("Attack detected and blocked probabilistically by defense. Reward:", reward)
                if not blocked:
                    overall_reward = 0
                    for outcome in self.cve_outcomes:
                        reward = self.reward_map.get(outcome.replace(" ", "_"), 0)
                        if self.verbose > 2:
                            print("Base reward for outcome", outcome, ":", reward)
                        # Outcome effects
                        if outcome in ['credential access', 'lateral move']:
                            if tgt_dict['compromised']:
                                reward = 0
                                self.repeated_actions_per_outcome[outcome] += 1
                            else:
                                self.outcomes_executed.append(outcome)
                                self.valid_actions_per_outcome[outcome] += 1
                            tgt_dict['compromised'] = True
                            tgt_dict['privilege_level'] = 'user'
                            self.compromised_nodes.add(tgt)
                            if self.verbose > 2:
                                print("Node", tgt, "compromised. Total compromised nodes:", len(self.compromised_nodes))
                            #    print("Final reward after compromise:", reward)
                        elif outcome == 'reconnaissance':
                            factor = self._reveal_neighbors(tgt, self.p_recon)
                            reward *= factor  # scale reward by number of nodes revealed
                            if factor > 0:
                                self.outcomes_executed.append(outcome)
                                self.valid_actions_per_outcome[outcome] += 1
                            else:
                                self.repeated_actions_per_outcome[outcome] += 1
                            if self.verbose > 2:
                                print("Revealed", factor, "neighbors of node", tgt, ". Final reward after reconnaissance:", reward)
                        elif outcome == 'discovery':
                            if tgt_dict['feature_visible']:
                                reward = 0  # no reward if already visible
                                self.repeated_actions_per_outcome[outcome] += 1
                            else:
                                self.outcomes_executed.append(outcome)
                                self.valid_actions_per_outcome[outcome] += 1
                            tgt_dict['feature_visible'] = True
                            if self.verbose > 2:
                                print("Node", tgt, "feature_visible set to True. Final reward after discovery:", reward)
                        elif outcome == 'collection':
                            if tgt_dict['data_present'] and not tgt_dict['data_collected']:
                                tgt_dict['data_collected'] = True
                                self.outcomes_executed.append(outcome)
                                self.valid_actions_per_outcome[outcome] += 1
                            else:
                                reward = 0
                                self.repeated_actions_per_outcome[outcome] += 1
                            if self.verbose > 2:
                                print("Node", tgt, "data_collected set to True. Final reward after collection:", reward)
                        elif outcome == 'exfiltration':
                            if tgt_dict['data_collected'] and not tgt_dict['data_exfiltrated']:
                                tgt_dict['data_exfiltrated'] = True
                                self.outcomes_executed.append(outcome)
                                self.valid_actions_per_outcome[outcome] += 1
                            else:
                                reward = 0
                                self.repeated_actions_per_outcome[outcome] += 1
                            if self.verbose > 2:
                                print("Node", tgt, "data_exfiltrated set to True. Final reward after exfiltration:", reward)
                        elif outcome == 'DOS':
                            if tgt_dict['dos']:
                                reward = 0
                                self.repeated_actions_per_outcome[outcome] += 1
                            else:
                                tgt_dict['dos'] = True
                                tgt_dict['graph_visible'] = False
                                if tgt in self.compromised_nodes:
                                    self.compromised_nodes.remove(tgt)
                                self.dos_nodes.add(tgt)
                                self.visible_nodes.remove(tgt)
                                self.outcomes_executed.append(outcome)
                                self.valid_actions_per_outcome[outcome] += 1
                            if self.verbose > 2:
                                print("Node", tgt, "set to DOS. Final reward after DOS:", reward)
                            assert not tgt_dict['graph_visible'], "Node in DOS state should not be visible."
                            assert tgt not in self.compromised_nodes, "Node in DOS state should not be compromised."
                            assert tgt not in self.visible_nodes, "Node in DOS state should not be in visible nodes."
                        elif outcome == 'persistence':
                            if tgt_dict['persistent']:
                                reward = 0
                                self.repeated_actions_per_outcome[outcome] += 1
                            else:
                                self.outcomes_executed.append(outcome)
                                self.valid_actions_per_outcome[outcome] += 1
                            tgt_dict['persistent'] = True
                            if self.verbose > 2:
                                print("Node", tgt, "persistent set to True. Final reward after persistence:", reward)
                        elif outcome == 'defense evasion':
                            if tgt_dict['defense_evaded']:
                                reward = 0
                                self.repeated_actions_per_outcome[outcome] += 1
                            else:
                                self.outcomes_executed.append(outcome)
                                self.valid_actions_per_outcome[outcome] += 1
                            tgt_dict['defense_evaded'] = True

                            if self.verbose > 2:
                                print("Node", tgt, "defense_evaded set to True. Final reward after defense evasion:", reward)
                        elif outcome == 'privilege escalation':
                            if tgt_dict['privilege_level'] == 'root':
                                reward = 0
                                self.repeated_actions_per_outcome[outcome] += 1
                            elif tgt_dict['privilege_level'] == 'none':
                                reward = 0
                                self.privilege_escalation_invalid_actions += 1
                                if self.verbose > 2:
                                    print("Node", tgt, "node not controlled, cannot escalate. Final reward after privilege escalation:", reward)
                            else:
                                self.outcomes_executed.append(outcome)
                                self.valid_actions_per_outcome[outcome] += 1
                                tgt_dict['privilege_level'] = 'root'
                                if self.verbose > 2:
                                    print("Node", tgt, "privilege_level set to root. Final reward after privilege escalation:", reward)

                        # Record exploited vuln description
                        if self.attack_graph.has_edge(src, tgt):
                            self.attack_graph.edges[src, tgt]['edge_attr_dict']['exploited_vulns'].append(self.cve_info['description'])
                        else:
                            self.attack_graph.add_edge(src, tgt)
                            self.attack_graph.edges[src, tgt]['edge_attr_dict'] = {}
                            self.attack_graph.edges[src, tgt]['edge_attr_dict']['exploited_vulns'] = [self.cve_info['description']]
                        self.attack_graph_edges += 1
                        if self.verbose > 2:
                            print("Updated attack graph edges with exploited vulnerability.")
                            print("Current pair", src, tgt, "has edges:", self.attack_graph.edges[src, tgt])
                        overall_reward += reward
                else:
                    overall_reward = reward
                self.reward = overall_reward
            else:
                self.reward = self.invalid_action_penalty_sum / (
                                min(self.max_steps_overall, self.max_steps_coefficient * self.n_nodes))

        # Step termination
        self.done = self.check_termination()
        self.action = (src, tgt, cve_id, outcome) if (self.outcome_selection or self.outcome_selection_all) else (src, tgt, cve_id)
        self.obs = self._get_obs()
        self.info = {}
        # update attack graph visible to the agent
        self.copy_true_into_attack_graph()
        super().post_step()
        return self.obs, self.reward, self.done, self.truncated, self.info

    def check_termination(self):
        if self.goal == 'control':
            if len(self.compromised_nodes) == self.n_nodes:
                return True
        # to potentially extend to other goals
        return False

    def update_metrics(self):
        super().update_metrics()
        compromised = len(self.compromised_nodes)
        visible = len(self.visible_nodes)
        total = self.n_nodes
        self._metrics.update({
            'compromised_nodes': compromised,
            'compromised_nodes_percentage': compromised / total if total > 0 else 0,
            'visible_nodes': visible,
            'visible_nodes_percentage': visible / total if total > 0 else 0,
            'DOS_nodes': len(self.dos_nodes),
            'DOS_nodes_percentage': len(self.dos_nodes) / total if total > 0 else 0,
            'total_nodes': total,
            'privilege_escalation_invalid_actions': self.privilege_escalation_invalid_actions,
            'privileges_invalid_actions': self.privileges_invalid_actions,
            'blocked_actions_defense': self.blocked_actions_defense,
            'attack_graph_edges': self.attack_graph_edges,
        })
        for outcome in self.outcomes:
            self._metrics[f'valid_actions_{outcome}'] = self.valid_actions_per_outcome[outcome]
            self._metrics[f'repeated_actions_{outcome}'] = self.repeated_actions_per_outcome[outcome]

class ExtendedCyberAttackEnv(CyberAttackEnv, ContinuousEnv):
    def __init__(self, **config):
        super().__init__(**config)
        self.update_spec()

    def update_spec(self, **kwargs): # reconstruction ignored, as it is mandatory
        self._observation_type = {
            'graph': Graph(poolings=[MeanPooling, SumPooling, MinPooling, MaxPooling], graph_name='attack_graph'),
            'nodes_number': Function(func=get_number_nodes, graph_name='attack_graph'),
            'edges_number': Function(func=get_number_edges, graph_name='attack_graph'),
            'average_node_degree': Function(func=get_average_node_degree, graph_name='attack_graph'),
            'graph_density': Function(func=get_graph_density, graph_name='attack_graph'),
            'action_space': ActionSpace(poolings=[MeanPooling, SumPooling, MaxPooling, MinPooling])
        }
        self.all_cves_descriptions = [cve['description'] for cve in self.all_cves]
        self.outcomes_per_cve = {
            cve['description']: cve['outcomes'] for cve in self.all_cves
        }
        if self.remove_invalid_actions:
            self._action_type = {
                'source_node': Node(graph_name='attack_graph', spec={'compromised': True, 'dos': False, 'graph_visible': True}),
                'target_node': Node(graph_name='attack_graph', spec={'graph_visible': True, 'dos': False}),
                'vulnerability': Object(name='vulnerability', set=self.all_cves_descriptions,
                                        reference_key='target_node',
                                        reference_feature_vector_key='vulns', reference_graph_name='attack_graph',
                                        feature_extractor=LanguageModelEmbedding,
                                        key_to_extract='description',
                                        feature_extractor_args={'model_name': 'bert-base-uncased',
                                                                'hf_key': self.huggingface['key']},
                                        embedding_size=768,
                                        caching=True)
            }
        else:
            self._action_type = {
                'source_node': Node(graph_name='attack_graph'),
                'target_node': Node(graph_name='attack_graph'),
                'vulnerability': Object(name='vulnerability', set=self.all_cves_descriptions, reference_key='target_node',
                                reference_feature_vector_key='vulns', reference_graph_name='attack_graph',
                                feature_extractor=LanguageModelEmbedding,
                                key_to_extract='description',
                                feature_extractor_args={'model_name': 'bert-base-uncased',
                                                        'hf_key':  self.huggingface['key']},
                                embedding_size=768,
                                caching=True)
            }
        if self.outcome_selection_all:
            # adding outcome selection
            self._action_type['outcome'] = Object(name='outcome',
                              set=self.outcomes,
                              # set_dict=self.outcomes_per_cve,
                              # set_dict_key_name='vulnerability',
                              feature_extractor=OneHotEncoding,
                              feature_extractor_args={'set': self.outcomes},
                              embedding_size=len(self.outcomes),
                              reference_graph_name='global')
        elif self.outcome_selection:
            self._action_type['outcome'] = Object(name='outcome',
                                                  set=self.outcomes,
                                                  set_dict=self.outcomes_per_cve,
                                                  set_dict_key_name='vulnerability',
                                                  feature_extractor=OneHotEncoding,
                                                  feature_extractor_args={'set': self.outcomes},
                                                  embedding_size=len(self.outcomes),
                                                  reference_graph_name='global')
        self._node_attributes = {
            'vulns': Attribute(feature_extractor=LanguageModelEmbedding,
                               feature_extractor_args={'model_name': 'bert-base-uncased', 'hf_key': self.huggingface['key']},
                               poolings=[SumPooling],
                               encoding=ContinuousValue(
                                   normalization="l2",
                               ),
                               embedding_size=768,
                               caching=True,
                               key_to_extract='description',
                               graph_name='attack_graph'), # vulns is a list
            'feature_visible': Attribute(feature_extractor=OneHotEncoding,
                                        feature_extractor_args={'set': [False, True]},
                                        encoding=BinaryValue(),
                                        embedding_size=2,
                                        graph_name='attack_graph'),
            'compromised': Attribute(feature_extractor=OneHotEncoding,
                                      feature_extractor_args={'set': [False, True]},
                                      encoding=BinaryValue(),
                                     embedding_size=2,
                                     graph_name='attack_graph'),
            'privilege_level': Attribute(feature_extractor=OneHotEncoding,
                                      feature_extractor_args={'set': ['none', 'user', 'root']},
                                      encoding=MultiClassValue(num_classes=3),
                                        embedding_size=3,
                                        graph_name='attack_graph'),
            'data_present': Attribute(feature_extractor=OneHotEncoding,
                                        feature_extractor_args={'set': [False, True]},
                                        encoding=BinaryValue(),
                                         embedding_size=2,
                                         graph_name='attack_graph'),
            'data_collected': Attribute(feature_extractor=OneHotEncoding,
                                        feature_extractor_args={'set': [False, True]},
                                        encoding=BinaryValue(),
                                         embedding_size=2,
                                         graph_name='attack_graph'),
            'data_exfiltrated': Attribute(feature_extractor=OneHotEncoding,
                                        feature_extractor_args={'set': [False, True]},
                                        encoding=BinaryValue(),
                                         embedding_size=2,
                                         graph_name='attack_graph'),
            'persistent': Attribute(feature_extractor=OneHotEncoding,
                                        feature_extractor_args={'set': [False, True]},
                                        encoding=BinaryValue(),
                                         embedding_size=2,
                                         graph_name='attack_graph'),
            'defense_evaded': Attribute(feature_extractor=OneHotEncoding,
                                        feature_extractor_args={'set': [False, True]},
                                        encoding=BinaryValue(),
                                         embedding_size=2,
                                         graph_name='attack_graph'),
        }
        self._edge_attributes = {
            'exploited_vulns': Attribute(feature_extractor=LanguageModelEmbedding,
                                feature_extractor_args={'model_name': 'bert-base-uncased', 'hf_key': self.huggingface['key']},
                               poolings=[SumPooling],
                               encoding=ContinuousValue(
                                   normalization="l2",
                                ),
                                embedding_size=768,
                                caching=True,
                               graph_name='attack_graph'), # vulns is a number integer
        }
        if self.no_action_support:
            self.discrete_actions = ["no_action"]
        else:
            self.discrete_actions = []

        self.action_candidates_reconstruction = True
        self.action_space_reconstruction_each_step = True
        self.reconstruct_only_changed_nodes = False

    def sample_valid_action(self):
        while True:
            src = random.choice(list(self.compromised_nodes))
            tgt = random.choice(list(self.visible_nodes))

            tgt_vulns = self.true_graph.nodes[tgt]["x_dict"]["vulns"]
            if not tgt_vulns:
                continue

            cve_info = random.choice(tgt_vulns)
            cve_id = cve_info["cve_id"]

            if self.outcome_selection or self.outcome_selection_all:
                if not cve_info["outcomes"]:
                    continue
                outcome = random.choice(cve_info["outcomes"])
                valid = self.is_valid_action(src, tgt, cve_id, outcome)
                if valid:
                    return (src, tgt, cve_id, outcome)
            else:
                valid = self.is_valid_action(src, tgt, cve_id)
                if valid:
                    return (src, tgt, cve_id)

    def get_graphs(self):
        return {"attack_graph": self.attack_graph}