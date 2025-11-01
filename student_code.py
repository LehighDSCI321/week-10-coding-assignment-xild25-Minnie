"""Student graph implementation for Week 10.

Implements:
- VersatileDigraph
- SortableDigraph
- TraversableDigraph
- DAG
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Iterator
from typing import Any

__all__ = [
    "VersatileDigraph",
    "SortableDigraph",
    "TraversableDigraph",
    "DAG",
]

Node = Hashable


class VersatileDigraph:
    """Minimal directed graph with adjacency, node values, and edge weights."""

    def __init__(self) -> None:
        # adjacency lists preserve insertion order by using lists
        self._succ: dict[Node, list[Node]] = {}
        self._pred: dict[Node, list[Node]] = {}
        # node values (from add_node value argument)
        self._value: dict[Node, Any] = {}
        # edge weights: (u, v) -> weight (from add_edge edge_weight argument)
        self._edge_weight: dict[tuple[Node, Node], Any] = {}

    def get_nodes(self) -> list[Node]:
        """Return all nodes as a list in insertion order."""
        return list(self._succ.keys())

    def get_node_value(self, u: Node) -> Any:
        """Return the stored value for node `u` (or None if not set)."""
        return self._value.get(u, None)

    def get_edge_weight(self, u: Node, v: Node) -> Any:
        """Return the stored weight for edge u->v (or None if not set)."""
        return self._edge_weight.get((u, v), None)

    def add_node(self, u: Node, value: Any | None = None) -> None:
        """Add node `u`. If `value` is provided, store/update it."""
        if u not in self._succ:
            self._succ[u] = []
            self._pred[u] = []
        if value is not None:
            self._value[u] = value

    def add_edge(self, u: Node, v: Node, edge_weight: Any | None = None) -> None:
        """Add a directed edge u->v (creates nodes if missing). Store edge_weight if given."""
        self.add_node(u)
        self.add_node(v)
        if v not in self._succ[u]:
            self._succ[u].append(v)
        if u not in self._pred[v]:
            self._pred[v].append(u)
        # store/overwrite edge weight (even if None — tests only check when provided)
        self._edge_weight[(u, v)] = edge_weight

    # ---- queries / helpers ----
    def nodes(self) -> Iterable[Node]:
        """Iterate nodes."""
        return self._succ.keys()

    def edges(self) -> Iterable[tuple[Node, Node]]:
        """Iterate edges (u, v) in insertion order."""
        for u, nbrs in self._succ.items():
            for v in nbrs:
                yield (u, v)

    def neighbors(self, u: Node) -> Iterable[Node]:
        """Alias for successors to match some callers."""
        return self.successors(u)

    def successors(self, u: Node) -> list[Node]:
        """Return successors of u as a list."""
        return list(self._succ.get(u, []))

    def predecessors(self, u: Node) -> list[Node]:
        """Return predecessors of u as a list."""
        return list(self._pred.get(u, []))

    def __contains__(self, u: Node) -> bool:
        return u in self._succ


class SortableDigraph(VersatileDigraph):
    """Adds a topological sort method named `top_sort` (Kahn's algorithm)."""

    def top_sort(self) -> list[Node]:
        """Return a topological order for DAGs; raise if cycle exists."""
        indeg: dict[Node, int] = {u: 0 for u in self.nodes()}
        for u, v in self.edges():
            indeg[v] += 1

        q: deque[Node] = deque([u for u, d in indeg.items() if d == 0])
        order: list[Node] = []

        while q:
            u = q.popleft()
            order.append(u)
            for v in self.successors(u):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(order) != len(indeg):
            raise ValueError("Graph has at least one cycle; cannot topologically sort.")
        return order


class TraversableDigraph(SortableDigraph):
    """Adds DFS and BFS traversals that EXCLUDE the start node from output."""

    def dfs(self, start: Node) -> Iterator[Node]:
        """Depth-first traversal (preorder) excluding `start`."""
        if start not in self:
            return
            yield  # keep as generator

        visited: set[Node] = {start}

        def _dfs(u: Node) -> Iterator[Node]:
            visited.add(u)
            yield u
            for w in self.successors(u):
                if w not in visited:
                    yield from _dfs(w)

        # exclude start: begin from its direct successors
        for v in self.successors(start):
            if v not in visited:
                yield from _dfs(v)

    def bfs(self, start: Node) -> Iterator[Node]:
        """Breadth-first traversal excluding `start`."""
        if start not in self:
            return
            yield

        visited: set[Node] = {start}
        q: deque[Node] = deque(self.successors(start))
        for v in q:
            visited.add(v)

        while q:
            u = q.popleft()
            yield u
            for w in self.successors(u):
                if w not in visited:
                    visited.add(w)
                    q.append(w)


class DAG(TraversableDigraph):
    """Directed acyclic graph that forbids edges which create cycles."""

    def _path_exists(self, src: Node, dst: Node) -> bool:
        """Return True iff a path exists from `src` to `dst`."""
        if src not in self or dst not in self:
            return False
        if src == dst:
            return True

        seen: set[Node] = set()
        stack: list[Node] = [src]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for w in self.successors(u):
                if w == dst:
                    return True
                if w not in seen:
                    stack.append(w)
        return False

    def add_edge(self, u: Node, v: Node, edge_weight: Any | None = None) -> None:
        """Add u->v unless it would create a cycle; otherwise raise ValueError."""
        # Ensure nodes exist so _path_exists can see them
        self.add_node(u)
        self.add_node(v)
        if u == v or self._path_exists(v, u):
            raise ValueError(f"Adding edge {u!r} -> {v!r} would create a cycle.")
        super().add_edge(u, v, edge_weight=edge_weight)
