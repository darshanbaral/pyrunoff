from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class Network:
    """Represent a directed network described by ordered node paths.

    :param paths: Paths whose consecutive nodes define directed edges.
    :param prevent_split: Whether to reject nodes with multiple downstream
        neighbors.
    :ivar sorted_nodes: Nodes in topological execution order.
    :ivar upstream_nodes: Direct upstream parents for each node.
    """

    paths: list[list[str]]
    prevent_split: bool = True
    sorted_nodes: list[str] = field(init=False)
    upstream_nodes: dict[str, set[str]] = field(init=False)

    def __post_init__(self):
        """Compute the topological order and upstream-node mapping."""
        self.sorted_nodes, self.upstream_nodes = self.toposort(
            self.paths, self.prevent_split
        )

    @staticmethod
    def toposort(paths, prevent_split: bool = True):
        """
        Sort nodes topologically and map each node to its direct upstream nodes.

        :param paths: Paths represented as ordered lists of node names.
        :param prevent_split: If true, reject nodes with multiple downstream
            neighbors.
        :return: A tuple containing the ordered nodes and upstream-node mapping.
        :raises ValueError: If a node diverges while splitting is disabled or
            the paths contain a cycle.
        """

        # Build a directed graph from the paths
        graph = defaultdict(list)
        in_degrees = defaultdict(int)
        nodes = set()
        upstream_nodes = defaultdict(set)

        for path in paths:
            nodes.update(path)

            if len(path) < 2:
                continue

            for i in range(1, len(path)):
                from_, to_ = path[i - 1], path[i]

                # Check if this specific edge is already recorded
                if from_ in upstream_nodes[to_]:
                    continue

                if prevent_split and len(graph[from_]) > 0 and graph[from_] != [to_]:
                    raise ValueError(f"Graph contains divergence at node {from_}")

                graph[from_].append(to_)
                upstream_nodes[to_].add(from_)
                in_degrees[to_] += 1

        # Kahn's Algorithm
        headwaters = sorted([node for node in nodes if in_degrees[node] == 0])
        queue = deque(headwaters)
        sorted_nodes = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for neighbor in graph[node]:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(sorted_nodes) != len(nodes):
            unprocessed_nodes = nodes - set(sorted_nodes)
            raise ValueError(
                f"Cycle detected or unreachable nodes: {unprocessed_nodes}"
            )

        full_upstream_nodes = {node: upstream_nodes.get(node, set()) for node in nodes}

        return sorted_nodes, full_upstream_nodes
