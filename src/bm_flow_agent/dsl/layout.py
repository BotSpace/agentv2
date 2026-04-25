from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


SPECIAL_TARGET_HANDLES = {"selected-message-target"}


def assign_positions(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    trigger_ids: list[str],
) -> dict[str, dict[str, int]]:
    node_ids = [node["id"] for node in nodes]
    indegree = {node_id: 0 for node_id in node_ids}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        if edge.get("targetHandle") in SPECIAL_TARGET_HANDLES:
            continue
        source = edge["source"]
        target = edge["target"]
        adjacency[source].append(target)
        indegree[target] = indegree.get(target, 0) + 1

    roots = [node_id for node_id in trigger_ids if node_id in indegree]
    roots.extend(sorted(node_id for node_id, degree in indegree.items() if degree == 0 and node_id not in roots))
    if not roots:
        roots = sorted(node_ids)

    depth: dict[str, int] = {}
    queue = deque((root, 0) for root in roots)
    while queue:
        node_id, node_depth = queue.popleft()
        current = depth.get(node_id)
        if current is not None and current <= node_depth:
            continue
        depth[node_id] = node_depth
        for neighbor in adjacency.get(node_id, []):
            queue.append((neighbor, node_depth + 1))

    for node_id in node_ids:
        depth.setdefault(node_id, 0)

    columns: dict[int, list[str]] = defaultdict(list)
    for node_id, node_depth in depth.items():
        columns[node_depth].append(node_id)

    positions: dict[str, dict[str, int]] = {}
    for column_index in sorted(columns):
        for row_index, node_id in enumerate(sorted(columns[column_index])):
            positions[node_id] = {"x": column_index * 520, "y": row_index * 240}

    return positions
