#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# Licensed under the MIT License.

'''
    cyberattack_utils.py
    This file contains supporting modules for the cyber attack path prediction benchmark.
'''

import random
import math

def sample_service_vuln_pairs(services_dict, n_vulns_per_node,
                              max_dos_percentage=0.25, max_lateral_percentage=0.25):
    """
    Guarantees:
      - exactly n_vulns_per_node per pair
      - at least ONE lateral move and ONE reconnaissance vuln per pair
      - balanced outcome diversity (not union coverage)
      - bounded number of DOS and lateral vulnerabilities per pair
    """

    service_buffers = []
    all_vulns_global = []

    max_dos = max(1, math.ceil(max_dos_percentage * n_vulns_per_node))
    max_lateral = max(1, math.ceil(max_lateral_percentage * n_vulns_per_node))

    # ---- PER-SERVICE COLLECTION ----
    for s in services_dict:
        vulns = s.get("vulns", {})
        if not vulns:
            continue

        collected = []
        for cve_id, cve_info in vulns.items():
            classes = [c["class"] for c in cve_info.get("classes", [])]
            if not classes:
                continue

            # Map credential access → lateral move
            if "credential access" in classes:
                classes = [c for c in classes if c != "credential access"]
                classes.append("lateral move")

            # Remove execution
            if "execution" in classes:
                classes.remove("execution")

            # Remove DOS if mixed and outcome_selection not enabled
            if "DOS" in classes and len(classes) > 1:
                classes.remove("DOS")

            # check that there should be at least one class
            if not classes:
                continue

            collected.append({
                "cve_id": cve_id,
                "product": s["product"],
                "version": s["version"],
                "outcomes": list(set(classes)),
                "description": cve_info.get("description", cve_id),
                "metrics": cve_info.get("metrics", {})
            })

        if not collected:
            continue

        random.shuffle(collected)

        service_buffers.append({
            "services": [{"product": s["product"], "version": s["version"]}],
            "vulns": collected
        })

        all_vulns_global.extend(collected)

    assert all_vulns_global, "No vulnerabilities available at all"
    assert any("reconnaissance" in v["outcomes"] for v in all_vulns_global), \
        "Dataset contains no reconnaissance vulnerabilities at all"

    pairs = []
    carry_services = []
    carry_vulns = []

    # ---- BUILD PAIRS ----
    for buf in service_buffers:
        vulns = buf["vulns"]
        services = buf["services"]

        cred_vulns = [v for v in vulns if "lateral move" in v["outcomes"]]
        recon_vulns = [v for v in vulns if "reconnaissance" in v["outcomes"]]

        while cred_vulns and len(vulns) >= n_vulns_per_node:
            # ---- SEED LATERAL AND RECON ----
            selected = [cred_vulns.pop(0)]

            if "reconnaissance" not in selected[0]["outcomes"] and recon_vulns:
                recon = random.choice(recon_vulns)
                if recon not in selected:
                    selected.append(recon)

            # ---- INIT COUNTERS ----
            outcome_counts = {}
            dos_count = 0
            lateral_count = sum("lateral move" in v["outcomes"] for v in selected)
            recon_count = sum("reconnaissance" in v["outcomes"] for v in selected)

            for v in selected:
                for o in v["outcomes"]:
                    outcome_counts[o] = outcome_counts.get(o, 0) + 1
                if "DOS" in v["outcomes"]:
                    dos_count += 1

            candidates = [v for v in vulns if v not in selected]

            # ---- FILL THE REST ----
            while len(selected) < n_vulns_per_node and candidates:
                best = max(
                    candidates,
                    key=lambda v: _diversity_score_limited(
                        v, outcome_counts, dos_count, max_dos,
                        lateral_count, max_lateral
                    )
                )

                if _diversity_score_limited(
                    best, outcome_counts, dos_count, max_dos,
                    lateral_count, max_lateral
                ) == -float("inf"):
                    candidates.remove(best)
                    continue

                candidates.remove(best)
                selected.append(best)

                for o in best["outcomes"]:
                    outcome_counts[o] = outcome_counts.get(o, 0) + 1
                if "DOS" in best["outcomes"]:
                    dos_count += 1
                if "lateral move" in best["outcomes"]:
                    lateral_count += 1
                if "reconnaissance" in best["outcomes"]:
                    recon_count += 1

            # ---- FALLBACK IF STILL SHORT ----
            while len(selected) < n_vulns_per_node:
                v = _safe_fallback_vuln_limited(
                    all_vulns_global, dos_count, max_dos,
                    lateral_count, max_lateral
                )
                assert v is not None, "Cannot satisfy DOS / lateral constraints"

                selected.append(v)
                for o in v["outcomes"]:
                    outcome_counts[o] = outcome_counts.get(o, 0) + 1
                if "DOS" in v["outcomes"]:
                    dos_count += 1
                if "lateral move" in v["outcomes"]:
                    lateral_count += 1
                if "reconnaissance" in v["outcomes"]:
                    recon_count += 1

            pairs.append({
                "services": services,
                "vulns": _finalize_chunk(selected)
            })

            for v in selected:
                if v in vulns:
                    vulns.remove(v)

        if vulns:
            carry_services.extend(services)
            carry_vulns.extend(vulns)

    # ---- HANDLE REMAINDER ----
    if carry_vulns:
        cred_pool = [v for v in carry_vulns if "lateral move" in v["outcomes"]]
        assert cred_pool, "Cannot form pair with credential access"

        selected = [random.choice(cred_pool)]

        # ---- INIT COUNTERS ----
        outcome_counts = {}
        dos_count = 0
        lateral_count = sum("lateral move" in v["outcomes"] for v in selected)
        recon_count = sum("reconnaissance" in v["outcomes"] for v in selected)
        for v in selected:
            for o in v["outcomes"]:
                outcome_counts[o] = outcome_counts.get(o, 0) + 1
            if "DOS" in v["outcomes"]:
                dos_count += 1

        remaining = [v for v in carry_vulns if v not in selected]

        while len(selected) < n_vulns_per_node and remaining:
            best = max(
                remaining,
                key=lambda v: _diversity_score_limited(
                    v, outcome_counts, dos_count, max_dos,
                    lateral_count, max_lateral
                )
            )
            remaining.remove(best)
            selected.append(best)

            for o in best["outcomes"]:
                outcome_counts[o] = outcome_counts.get(o, 0) + 1
            if "DOS" in best["outcomes"]:
                dos_count += 1
            if "lateral move" in best["outcomes"]:
                 lateral_count += 1
            if "reconnaissance" in best["outcomes"]:
                recon_count += 1

        # ---- FALLBACK ----
        while len(selected) < n_vulns_per_node:
            v = _safe_fallback_vuln_limited(
                all_vulns_global, dos_count, max_dos,
                lateral_count, max_lateral
            )
            selected.append(v)
            for o in v["outcomes"]:
                outcome_counts[o] = outcome_counts.get(o, 0) + 1
            if "DOS" in v["outcomes"]:
                dos_count += 1
            if "lateral move" in v["outcomes"]:
                lateral_count += 1
            if "reconnaissance" in v["outcomes"]:
                recon_count += 1

        pairs.append({
            "services": carry_services.copy(),
            "vulns": _finalize_chunk(selected)
        })

    # ---- FINAL ASSERTS ----
    pairs_success = 0
    for p in pairs:
        assert len(p["vulns"]) == n_vulns_per_node
        lateral_count = sum("lateral move" in v["outcomes"] for v in p["vulns"])
        recon_count = sum("reconnaissance" in v["outcomes"] for v in p["vulns"])
        dos_count = sum("DOS" in v["outcomes"] for v in p["vulns"])

        if not ( lateral_count >= 1 and recon_count >= 1 and dos_count <= max_dos and lateral_count <= max_lateral ):
            return False

    random.shuffle(pairs)
    return pairs


def _diversity_score_limited(v, outcome_counts, dos_count, max_dos, lateral_count, max_lateral):
    # DOS / lateral max constraints
    if "DOS" in v["outcomes"] and dos_count >= max_dos:
        return -float("inf")
    if "lateral move" in v["outcomes"] and lateral_count >= max_lateral:
        return -float("inf")
    # reward novelty
    new_outcomes = set(v["outcomes"]) - set(outcome_counts.keys())
    # penalize overrepresented outcomes
    penalty = sum(outcome_counts.get(o, 0) for o in v["outcomes"])
    return len(new_outcomes) - 0.5 * penalty

def _safe_fallback_vuln_limited(all_vulns, dos_count, max_dos, lateral_count, max_lateral):
    # return a random vuln that satisfies the DOS and lateral constraints, or None if impossible
    candidates = [
        v for v in all_vulns
        if ("DOS" not in v["outcomes"] or dos_count < max_dos)
        and ("lateral move" not in v["outcomes"] or lateral_count < max_lateral)
    ]
    if candidates:
        return random.choice(candidates)
    return None



def _finalize_chunk(chunk):
     # Convert to final format and remove any duplicates that might have been introduced by the fallback mechanism
    return [{
        "cve_id": v["cve_id"],
        "product": v["product"],
        "version": v["version"],
        "outcomes": v["outcomes"],
        "description": v.get("description"),
        "metrics": v.get("metrics", {}),
    } for v in chunk]


def assign_pairs_to_nodes(
    service_vuln_pairs,
    n_nodes,
    overlap_probability
):
    # Function to assign service-vulnerability pairs to nodes, with controlled overlap
    assigned = []
    used_pairs = []
    pool_idx = 0

    for _ in range(n_nodes):
        reuse = (
            used_pairs
            and random.random() < overlap_probability
        )

        if reuse:
            pair = random.choice(used_pairs)
        else:
            if pool_idx >= len(service_vuln_pairs):
                pair = random.choice(used_pairs)
            else:
                pair = service_vuln_pairs[pool_idx]
                pool_idx += 1

        used_pairs.append(pair)
        assigned.append(pair)

    return assigned


