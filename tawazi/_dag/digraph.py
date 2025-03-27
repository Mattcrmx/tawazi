"""Module containing the definition of a Directed Graph Extension of networkx.DiGraph."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Any, Optional

import networkx as nx
from networkx import NetworkXNoCycle, NetworkXUnfeasible, find_cycle
from typing_extensions import Self

from tawazi.config import Config
from tawazi.consts import Identifier, Tag
from tawazi.errors import TawaziUsageError
from tawazi.node import ExecNode, UsageExecNode


@dataclass
class GraphNode:
    id: str
    debug: bool = False
    setup: bool = False
    priority: int = 0
    in_degree: int = 0
    out_degree: int = 0
    successors: Optional[list[str]] = None
    predecessors: Optional[list[str]] = None
    tag: Optional[Tag] = None

    @classmethod
    def from_xn(cls, xn: ExecNode):
        if xn.tag:
            if isinstance(xn.tag, Tag):
                tag = [xn.tag]
            else:
                tag = [t for t in xn.tag]
        else:
            tag = None

        predecessors = [xn.id for xn in xn.dependencies]

        return cls(
            id=xn.id,
            debug=xn.debug,
            setup=xn.setup,
            priority=xn.priority,
            predecessors=predecessors,
            in_degree=len(predecessors),
            tag=tag
        )


class DiGraph:
    def __init__(self) -> None:
        # Access to nodes data is very slow, using dict is faster
        self.tag: dict[Identifier, Optional[list[Tag]]] = defaultdict(lambda: None)
        self.debug: dict[Identifier, bool] = defaultdict(lambda: False)
        self.setup: dict[Identifier, bool] = defaultdict(lambda: False)
        self.compound_priority: dict[Identifier, int] = defaultdict(lambda: 0)
        self._graph = {}
        self.n_nodes = 0
        self.nodes: dict[str, GraphNode] = {}

    @property
    def debug_nodes(self) -> list[Identifier]:
        """Get the debug nodes.

        Returns:
            the debug nodes
        """
        return [id_ for id_, debug in self.debug.items() if debug]

    @property
    def setup_nodes(self) -> list[Identifier]:
        """Get the setup nodes.

        Returns:
            the setup nodes
        """
        return [id_ for id_, setup in self.setup.items() if setup]

    @property
    def tags(self) -> set[str]:
        """Get all the tags available for the graph.

        Returns:
            A set of tags
        """
        return set(chain(*(tags for tags in self.tag.values())))  # type: ignore[arg-type]

    @property
    def root_nodes(self) -> set[Identifier]:
        """Safely gets the root nodes.

        Returns:
            List of root nodes
        """
        return {node.id for node in self.nodes.values() if node.in_degree == 0}


    @property
    def leaf_nodes(self) -> list[Identifier]:
        """Safely get the leaf nodes.

        Returns:
            List of leaf nodes
        """
        return [node.id for node in self.nodes.values() if node.out_degree == 0]

    def _add_xn(self, xn: ExecNode, dag_input_ids: list[str]) -> None:
        # validate setup ExecNodes
        if xn.setup and any(dep.id in dag_input_ids for dep in xn.dependencies):
            raise TawaziUsageError(
                f"The ExecNode {xn} takes as parameters one of the DAG's input parameter"
            )

        # Create node from ExecNode
        node = GraphNode.from_xn(xn)
        self.nodes[node.id] = node
        self.setup[node.id] = node.setup
        self.debug[node.id] = node.debug
        self.compound_priority[node.id] = node.priority

        # Ensure an entry exists for the node in _graph
        if node.id not in self._graph:
            self._graph[node.id] = {"in": set(), "out": set()}
        # Add all predecessor edges: update the node's incoming edges
        self._graph[node.id]["in"].update(node.predecessors or [])

        # For each predecessor, add an outgoing edge to node
        for pred in node.predecessors or []:
            if pred not in self._graph:
                self._graph[pred] = {"in": set(), "out": set()}
            self._graph[pred]["out"].add(node.id)
            # If the predecessor node exists in our nodes dict, update its out_degree
            if pred in self.nodes:
                pred_node = self.nodes[pred]
                pred_node.out_degree = len(self._graph[pred]["out"])

        # Finally, update this node's degrees based on current graph structure.
        node.in_degree = len(self._graph[node.id]["in"])
        node.out_degree = len(self._graph[node.id]["out"])


    def _find_cycle(self):
        """
        Detects if there is a cycle in the directed graph.

        Returns:
            tuple: A tuple containing:
                - bool: True if a cycle is found, False otherwise
        """
        # Track the state of nodes during DFS
        not_visited = 0
        visiting = 1
        visited = 2

        node_states = {node_id: not_visited for node_id in self._graph}
        parent = {}

        def dfs(node_id: str):
            node_states[node_id] = visiting

            for neighbor in self._graph[node_id]['out']:
                # If neighbor is currently being visited, we found a cycle
                if node_states[neighbor] == visiting:
                    # Reconstruct the cycle path
                    cycle_path = [neighbor, node_id]
                    current = node_id
                    while parent.get(current) and parent[current] != neighbor:
                        current = parent[current]
                        cycle_path.append(current)
                    cycle_path.reverse()
                    return True, cycle_path

                # If neighbor is unvisited, explore it
                if node_states[neighbor] == not_visited:
                    parent[neighbor] = node_id
                    cycle_found, cycle_path = dfs(neighbor)
                    if cycle_found:
                        return True, cycle_path

            # mark node as completely visited
            node_states[node_id] = visited
            return False, []

        # multi source dfs
        for node_id in self._graph:
            if node_states[node_id] == not_visited:
                found, cycle = dfs(node_id)
                if found:
                    return True, cycle

        return False, []

    def _remove_node(self, node: str) -> None:
        # prune out and in degrees and deps
        for pred in self._graph[node].predecessors:
            self._graph[pred].sucessors.remove(node)
            self._graph[pred].out_degree -= 1

        for succ in self._graph[node].successors:
            self._graph[succ].predecessors.remove(node)
            self._graph[succ].in_degree -= 1

        del self._graph[node]


    def _dft(self, node: str) -> list[str]:
        """Simple depth first traversal.

        Args:
            node: the node from which to traverse

        Returns:
            the successors of the node.
        """
        def __dft(root, res):
            res.append(root)
            for succ in self._graph[node]["out"]:
                __dft(succ, res)

        result = []
        __dft(node, result)
        return result


    def single_node_successors(self, node_id: Identifier) -> list[Identifier]:
        """Get all the successors of a node with a depth first search.

        Args:
            node_id: the node acting as the root of the search

        Returns:
            list of the node's successors
        """
        return self._dft(node_id)

    def multiple_nodes_successors(self, nodes_ids: Sequence[Identifier]) -> set[Identifier]:
        """Get the successors of all nodes in the iterable.

        Args:
            nodes_ids: nodes of which we want successors

        Returns:
            a set of all the successors
        """
        return set(list(chain(*[self.single_node_successors(node_id) for node_id in nodes_ids])))


    def assign_compound_priority(self) -> None:
        """Assigns a compound priority to all nodes in the graph.

        The compound priority is the sum of the priorities of all children recursively.
        """
        # 1. start from bottom up
        leaf_ids = set(self.leaf_nodes)

        # 2. assign the compound priority for all the remaining nodes in the graph:
        # Priority assignment happens by epochs:
        # 2.1. during every epoch, we assign the compound priority for the parents of the current leaf nodes

        while leaf_ids:
            next_leaf_ids = set()
            for leaf_id in leaf_ids:
                compound_priority = self.compound_priority[leaf_id]

                # for parent nodes, this loop won't execute
                for parent_id in self.nodes[leaf_id].predecessors:
                    # increment the compound_priority of the parent node by the leaf priority
                    self.compound_priority[parent_id] += compound_priority

                    next_leaf_ids.add(parent_id)
            leaf_ids = next_leaf_ids

    def remove_root_node(self, root_node: Identifier) -> set[Identifier]:
        """Removes a root node and returns all the new root nodes generated by this modification to the DAG."""
        generated_root_nodes = {
            new_root_node
            for new_root_node in self._graph[root_node]["out"]
            if self.nodes[new_root_node].in_degree == 1
        }
        self._remove_node(root_node)
        return generated_root_nodes

    def remove_any_root_node(self) -> Identifier:
        """Removes any root node and returns the removed root node."""
        for node in self.root_nodes:
            self._remove_node(node)
            return node # type: ignore[no-any-return]
        raise ValueError("No root node to remove.")

    @classmethod
    def from_exec_nodes(
        cls, input_nodes: list[UsageExecNode], exec_nodes: dict[Identifier, ExecNode]
    ) -> Self:
        """Build a DigraphEx from exec nodes.

        Args:
            input_nodes: nodes that are the inputs of the graph
            exec_nodes: the graph nodes

        Returns:
            the DigraphEx object
        """
        graph = DiGraph()

        input_ids = [uxn.id for uxn in input_nodes]
        for xn in exec_nodes.values():
            graph._add_xn(xn, input_ids)

        # check for circular dependencies
        has_cycle, cycle = graph._find_cycle()

        if has_cycle:
            raise TawaziUsageError(f"the DAG contains at least a circular dependency: {cycle}")

        # compute the sum of priorities of all recursive children
        graph.assign_compound_priority()

        return graph


    def get_tagged_nodes(self, tag: Tag) -> list[str]:
        """Get nodes with a certain tag.

        Args:
            tag: the tag identifier

        Returns:
            a list of nodes
        """
        return [xn for xn, tags in self.tag.items() if tags is not None and tag in tags]


    def get_single_reachable_leaves(self, root: Identifier) -> list[Identifier]:
        """Get reachable leaf nodes from specified root.

        Args:
            root: the start node to consider

        Returns:
            the leaves reachable from that node.
        """
        reachable_leaves = self.single_node_successors(root)
        leaves = []

        for leaf in self.leaf_nodes:
            if leaf in reachable_leaves:
                leaves.append(leaf)

        return leaves

    def get_multiple_reachable_leaves(self, roots: list[Identifier]) -> list[Identifier]:
        """Get all reachable leaf nodes from specified roots.

        Args:
            roots: the start nodes to consider

        Returns:
            the leaves reachable from all roots.
        """
        all_reachable = []

        for root in roots:
            reachable_leaves = self.get_single_reachable_leaves(root)

            for leaf in reachable_leaves:
                if leaf not in all_reachable:
                    all_reachable.append(leaf)

        return all_reachable



class DiGraphEx(nx.DiGraph):
    """Extends the DiGraph with some methods."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Access to nodes data is very slow, using dict is faster
        self.tag: dict[Identifier, Optional[list[Tag]]] = defaultdict(lambda: None)
        self.debug: dict[Identifier, bool] = defaultdict(lambda: False)
        self.setup: dict[Identifier, bool] = defaultdict(lambda: False)
        self.compound_priority: dict[Identifier, int] = defaultdict(lambda: 0)

    @classmethod
    def from_exec_nodes(
        cls, input_nodes: list[UsageExecNode], exec_nodes: dict[Identifier, ExecNode]
    ) -> Self:
        """Build a DigraphEx from exec nodes.

        Args:
            input_nodes: nodes that are the inputs of the graph
            exec_nodes: the graph nodes

        Returns:
            the DigraphEx object
        """
        graph = DiGraphEx()

        input_ids = [uxn.id for uxn in input_nodes]
        for node in exec_nodes.values():
            # add node and edges
            graph.add_exec_node(node)

            # add tag, debug, setup and priority
            if node.tag:
                if isinstance(node.tag, Tag):
                    graph.tag[node.id] = [node.tag]
                else:
                    graph.tag[node.id] = [t for t in node.tag]

            graph.debug[node.id] = node.debug
            graph.setup[node.id] = node.setup
            graph.compound_priority[node.id] = node.priority

            # validate setup ExecNodes
            if node.setup and any(dep.id in input_ids for dep in node.dependencies):
                raise TawaziUsageError(
                    f"The ExecNode {node} takes as parameters one of the DAG's input parameter"
                )

        # check for circular dependencies
        try:
            cycle = find_cycle(graph)
            raise NetworkXUnfeasible(f"the DAG contains at least a circular dependency: {cycle}")
        except NetworkXNoCycle:
            pass

        # compute the sum of priorities of all recursive children
        graph.assign_compound_priority()

        return graph

    def add_exec_node(self, xn: ExecNode) -> None:
        """Add an ExecNode and its dependencies to the graph."""
        self.add_node(xn.id)
        self.add_edges_from([(dep.id, xn.id) for dep in xn.dependencies])

    def make_subgraph(
        self,
        target_nodes: Optional[list[str]] = None,
        exclude_nodes: Optional[list[str]] = None,
        root_nodes: Optional[list[str]] = None,
    ) -> Self:
        """Builds the DigraphEx, with potential graph pruning.

        Args:
            target_nodes: nodes that we want to run and their dependencies
            exclude_nodes: nodes that should be excluded from the graph
            root_nodes: base ancestor nodes from which to start graph resolution

        Returns:
            Base Graph that will be used for the computations
        """
        graph = deepcopy(self)

        # first try to heavily prune removing roots
        if root_nodes is not None:
            if not set(root_nodes).issubset(set(graph.root_nodes)):
                raise ValueError(
                    f"nodes {set(graph.root_nodes).difference(set(root_nodes))} aren't root nodes."
                )

            # extract subgraph with only provided roots
            # NOTE: copy is because edges/nodes are shared with original graph
            graph = graph.subgraph(graph.multiple_nodes_successors(root_nodes)).copy()

        # then exclude nodes
        if exclude_nodes is not None:
            graph.remove_nodes_from(graph.multiple_nodes_successors(exclude_nodes))

        # lastly select additional nodes
        if target_nodes is not None:
            graph = graph.minimal_induced_subgraph(target_nodes).copy()

        return graph

    @property
    def root_nodes(self) -> set[Identifier]:
        """Safely gets the root nodes.

        Returns:
            List of root nodes
        """
        return {node for node, degree in self.in_degree if degree == 0}

    def remove_root_node(self, root_node: Identifier) -> set[Identifier]:
        """Removes a root node and returns all the new root nodes generated by this modification to the DAG."""
        generated_root_nodes = {
            new_root_node
            for new_root_node in self.successors(root_node)
            if self.in_degree[new_root_node] == 1
        }
        self.remove_node(root_node)
        return generated_root_nodes

    def remove_any_root_node(self) -> Identifier:
        """Removes any root node and returns the removed root node."""
        for node, degree in self.in_degree:
            if degree == 0:
                self.remove_node(node)
                return node  # type: ignore[no-any-return]
        raise ValueError("No root node to remove.")

    @property
    def leaf_nodes(self) -> list[Identifier]:
        """Safely get the leaf nodes.

        Returns:
            List of leaf nodes
        """
        return [node for node, degree in self.out_degree if degree == 0]

    def get_single_reachable_leaves(self, root: Identifier) -> list[Identifier]:
        """Get reachable leaf nodes from specified root.

        Args:
            root: the start node to consider

        Returns:
            the leaves reachable from that node.
        """
        reachable_leaves = self.single_node_successors(root)
        leaves = []

        for leaf in self.leaf_nodes:
            if leaf in reachable_leaves:
                leaves.append(leaf)

        return leaves

    def get_multiple_reachable_leaves(self, roots: list[Identifier]) -> list[Identifier]:
        """Get all reachable leaf nodes from specified roots.

        Args:
            roots: the start nodes to consider

        Returns:
            the leaves reachable from all roots.
        """
        all_reachable = []

        for root in roots:
            reachable_leaves = self.get_single_reachable_leaves(root)

            for leaf in reachable_leaves:
                if leaf not in all_reachable:
                    all_reachable.append(leaf)

        return all_reachable

    @property
    def debug_nodes(self) -> list[Identifier]:
        """Get the debug nodes.

        Returns:
            the debug nodes
        """
        return [id_ for id_, debug in self.debug.items() if debug]

    @property
    def setup_nodes(self) -> list[Identifier]:
        """Get the setup nodes.

        Returns:
            the setup nodes
        """
        return [id_ for id_, setup in self.setup.items() if setup]

    @property
    def tags(self) -> set[str]:
        """Get all the tags available for the graph.

        Returns:
            A set of tags
        """
        return set(chain(*(tags for tags in self.tag.values())))  # type: ignore[arg-type]

    @property
    def topologically_sorted(self) -> list[Identifier]:
        """Makes the simple topological sort of the graph nodes.

        Returns:
            List of nodes of the graph listed in topological order
        """
        return list(nx.topological_sort(self))

    def get_tagged_nodes(self, tag: Tag) -> list[str]:
        """Get nodes with a certain tag.

        Args:
            tag: the tag identifier

        Returns:
            a list of nodes
        """
        return [xn for xn, tags in self.tag.items() if tags is not None and tag in tags]

    def single_node_successors(self, node_id: Identifier) -> list[Identifier]:
        """Get all the successors of a node with a depth first search.

        Args:
            node_id: the node acting as the root of the search

        Returns:
            list of the node's successors
        """
        return list(nx.dfs_tree(self, node_id).nodes())

    def multiple_nodes_successors(self, nodes_ids: Sequence[Identifier]) -> set[Identifier]:
        """Get the successors of all nodes in the iterable.

        Args:
            nodes_ids: nodes of which we want successors

        Returns:
            a set of all the sucessors
        """
        return set(list(chain(*[self.single_node_successors(node_id) for node_id in nodes_ids])))

    def remove_recursively(self, root_node: Identifier, remove_root_node: bool = True) -> None:
        """Recursively removes all the nodes that depend on the provided.

        Args:
            root_node (Identifier): the root node
            remove_root_node (bool, optional): whether to remove the root node or not. Defaults to True.
        """
        nodes_to_remove = self.single_node_successors(root_node)

        # skip removing the root node if requested
        if not remove_root_node:
            nodes_to_remove.remove(root_node)

        for node in nodes_to_remove:
            self.remove_node(node)

    def include_debug_nodes(self, leaves_ids: list[Identifier]) -> list[Identifier]:
        """Get debug nodes that are runnable with provided nodes as direct roots.

        For example:
        A
        |
        B
        | \
        D E

        if D is not a debug ExecNode and E is a debug ExecNode.
        If the subgraph whose leaf ExecNode D is executed,
        E should also be included in the execution because it can be executed (debug node whose inputs are provided)
        Hence we should extend the subgraph containing only D to also contain E

        Args:
            leaves_ids: the leaves ids of the subgraph

        Returns:
            the leaves ids of the new extended subgraph that contains more debug ExecNodes
        """
        new_debug_xn_discovered = True
        while new_debug_xn_discovered:
            new_debug_xn_discovered = False
            for id_ in leaves_ids:
                for successor_id in self.successors(id_):
                    if successor_id not in leaves_ids and successor_id in self.debug_nodes:
                        # a new debug XN has been discovered!
                        if set(self.predecessors(successor_id)).issubset(set(leaves_ids)):
                            new_debug_xn_discovered = True
                            # this new XN can run by only running the current leaves_ids
                            leaves_ids.append(successor_id)
        return leaves_ids

    def extend_graph_with_debug_nodes(self, original_graph: Self, cfg: Config) -> Self:
        """Add or remove debug nodes depending on the configuration.

        Args:
            original_graph: the graph containing a broader set of nodes
            cfg: the tawazi configuration

        Returns:
            the correct subgraph
        """
        if cfg.RUN_DEBUG_NODES:
            nodes_to_include = original_graph.include_debug_nodes(self.leaf_nodes) + list(
                self.nodes
            )
        else:
            nodes_to_include = list(set(self.nodes) - set(self.debug_nodes))

        # TODO: optimize this part by avoiding the copy of the graph which costing 40ms for findoc currently
        #  it can be done by reconstructing the graph instead of destroying it in the scheduler
        # networkx typing problem
        new_graph = original_graph.subgraph(nodes_to_include).copy()
        new_graph.tag = self.tag
        new_graph.debug = self.debug
        new_graph.setup = self.setup
        new_graph.compound_priority = self.compound_priority
        return new_graph  # type: ignore[no-any-return]

    def minimal_induced_subgraph(self, nodes: list[Identifier]) -> Self:
        """Get the minimal induced subgraph containing the provided nodes.

        The generated subgraph contains the provided nodes as leaf nodes.
        For example:
        graph =
        "
        A
        | \
        B  C
        |  |\
        D  E F
        "
        subgraph_leaves(D, C, E) ->
        "
        A
        | \
        B  C
        |  |
        D  E
        "
        C is not a node that can be made into leaf nodes

        Args:
            nodes: the list of nodes to be executed

        Returns:
            the induced subgraph over the original graph with the provided nodes:

        Raises:
            ValueError: if the provided nodes are not in the graph
        """
        if any([node not in self.nodes for node in nodes]):
            raise ValueError(
                f"The provided nodes are not in the graph. "
                f"The provided nodes are: {nodes}."
                f"The graph only contains: {self.nodes}."
            )

        # compute all the ancestor nodes that will be included in the graph
        all_ancestors = self.ancestors_of_iter(nodes)
        induced_subgraph: DiGraphEx = nx.induced_subgraph(self, all_ancestors | set(nodes))
        return induced_subgraph

    def ancestors_of_iter(self, nodes: Iterable[Identifier]) -> set[Identifier]:
        """Returns the ancestors of the provided nodes.

        Args:
            nodes (Set[Identifier]): The nodes to find the ancestors of

        Returns:
            Set[Identifier]: The ancestors of the provided nodes
        """
        return set().union(*chain(nx.ancestors(G=self, source=node) for node in nodes))

    def assign_compound_priority(self) -> None:
        """Assigns a compound priority to all nodes in the graph.

        The compound priority is the sum of the priorities of all children recursively.
        """
        # 1. start from bottom up
        leaf_ids = set(self.leaf_nodes)

        # 2. assign the compound priority for all the remaining nodes in the graph:
        # Priority assignment happens by epochs:
        # 2.1. during every epoch, we assign the compound priority for the parents of the current leaf nodes

        while leaf_ids:
            next_leaf_ids = set()
            for leaf_id in leaf_ids:
                compound_priority = self.compound_priority[leaf_id]

                # for parent nodes, this loop won't execute
                for parent_id in self.predecessors(leaf_id):
                    # increment the compound_priority of the parent node by the leaf priority
                    self.compound_priority[parent_id] += compound_priority

                    next_leaf_ids.add(parent_id)
            leaf_ids = next_leaf_ids
