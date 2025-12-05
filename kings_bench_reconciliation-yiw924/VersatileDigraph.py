# VersatileDigraph.py
# Clean directed graph implementation for network analysis

import json
from typing import Dict, List, Set, Tuple, Optional, Any
import networkx as nx

class VersatileDigraph:
    def __init__(self):
        # Node storage: {node_id: numeric_value}
        self._node_value = {}
        # Edge storage: {from: {to: {"weight": w, "name": name}}}
        self._edges = {}

    # =======================
    #        NODES
    # =======================
    def add_node(self, node_id: str, value: float = 0) -> None:
        """Add node if missing."""
        if node_id not in self._node_value:
            self._node_value[node_id] = value
            self._edges[node_id] = {}

    def nodes(self) -> List[str]:
        """Get all node IDs."""
        return list(self._node_value.keys())

    def get_node_value(self, node_id: str) -> float:
        """Get value of a node."""
        return self._node_value.get(node_id, 0)

    def set_node_value(self, node_id: str, value: float) -> None:
        """Set value of a node."""
        if node_id in self._node_value:
            self._node_value[node_id] = value

    # =======================
    #        EDGES
    # =======================
    def add_edge(self, start: str, end: str, edge_weight: float = 1, 
                 name: Optional[str] = None) -> None:
        """Add directed edge if nodes exist."""
        if start not in self._node_value:
            self.add_node(start, 0)
        if end not in self._node_value:
            self.add_node(end, 0)

        if name is None:
            name = f"edge_{len(self._edges[start]) + 1}"

        # If edge exists, increment weight
        if end in self._edges[start]:
            self._edges[start][end]["weight"] += edge_weight
        else:
            self._edges[start][end] = {"weight": edge_weight, "name": name}

    def neighbors(self, node_id: str) -> List[str]:
        """Get outgoing neighbors of a node."""
        return list(self._edges.get(node_id, {}).keys())

    def in_neighbors(self, node_id: str) -> List[str]:
        """Get incoming neighbors of a node."""
        incoming = []
        for source in self._edges:
            if node_id in self._edges[source]:
                incoming.append(source)
        return incoming

    def get_edge_weight(self, start: str, end: str) -> float:
        """Get weight of an edge."""
        return self._edges.get(start, {}).get(end, {}).get("weight", 0)

    def degree(self, node_id: str) -> int:
        """Get total degree (in + out) of a node."""
        out_degree = len(self.neighbors(node_id))
        in_degree = len(self.in_neighbors(node_id))
        return out_degree + in_degree

    def degree_centrality(self) -> Dict[str, float]:
        """Calculate degree centrality for all nodes."""
        n = len(self._node_value)
        if n <= 1:
            return {node: 0 for node in self.nodes()}
        
        centrality = {}
        for node in self.nodes():
            centrality[node] = self.degree(node) / (n - 1)
        
        return centrality

    # =======================
    #   NETWORK ANALYSIS
    # =======================
    def find_power_brokers(self, top_n: int = 10) -> List[Tuple[str, int, float]]:
        """Find nodes with highest degree centrality (Power Brokers)."""
        centrality = self.degree_centrality()
        results = []
        
        for node, cent in sorted(centrality.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:top_n]:
            deg = self.degree(node)
            results.append((node, deg, cent))
        
        return results

    def get_ego_network(self, center: str, radius: int = 1) -> Dict[str, Any]:
        """Extract ego network around a center node."""
        if center not in self._node_value:
            return {"nodes": [], "edges": [], "center": center}
        
        # BFS to get nodes within radius
        visited = {center: 0}
        queue = [(center, 0)]
        
        while queue:
            node, distance = queue.pop(0)
            if distance >= radius:
                continue
            
            # Add outgoing neighbors
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited[neighbor] = distance + 1
                    queue.append((neighbor, distance + 1))
            
            # Add incoming neighbors
            for neighbor in self.in_neighbors(node):
                if neighbor not in visited:
                    visited[neighbor] = distance + 1
                    queue.append((neighbor, distance + 1))
        
        # Extract edges between visited nodes
        edges = []
        for source in visited:
            for target in self.neighbors(source):
                if target in visited:
                    edge_info = self._edges[source][target]
                    edges.append({
                        "source": source,
                        "target": target,
                        "weight": edge_info["weight"],
                        "name": edge_info["name"]
                    })
        
        return {
            "nodes": list(visited.keys()),
            "edges": edges,
            "center": center,
            "radius": radius,
            "node_count": len(visited),
            "edge_count": len(edges)
        }

    # =======================
    #   CONVERSION METHODS
    # =======================
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph."""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node, value in self._node_value.items():
            G.add_node(node, weight=value)
        
        # Add edges
        for source in self._edges:
            for target, info in self._edges[source].items():
                G.add_edge(source, target, 
                          weight=info["weight"],
                          name=info["name"])
        
        return G

    def to_undirected_networkx(self) -> nx.Graph:
        """Convert to undirected NetworkX graph."""
        G = nx.Graph()
        
        # Add nodes with attributes
        for node, value in self._node_value.items():
            G.add_node(node, weight=value)
        
        # Add edges (combine weights for bidirectional edges)
        edge_weights = {}
        for source in self._edges:
            for target, info in self._edges[source].items():
                pair = tuple(sorted([source, target]))
                current = edge_weights.get(pair, 0)
                edge_weights[pair] = current + info["weight"]
        
        for (u, v), weight in edge_weights.items():
            G.add_edge(u, v, weight=weight)
        
        return G

    # =======================
    #   EXPORT METHODS
    # =======================
    def export_json(self, filename: str = "graph_data.json") -> None:
        """Export graph data to JSON file."""
        data = {
            "nodes": [
                {"id": node, "value": value} 
                for node, value in self._node_value.items()
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "weight": info["weight"],
                    "name": info["name"]
                }
                for source in self._edges
                for target, info in self._edges[source].items()
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[OK] Graph data exported to {filename}")

    def export_gexf(self, filename: str = "graph.gexf") -> None:
        """Export to GEXF format for Gephi visualization."""
        try:
            G = self.to_networkx()
            nx.write_gexf(G, filename)
            print(f"[OK] GEXF file exported to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to export GEXF: {e}")

    # =======================
    #   STATISTICS
    # =======================
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic graph statistics."""
        total_edges = sum(len(edges) for edges in self._edges.values())
        total_weight = sum(
            info["weight"]
            for edges in self._edges.values()
            for info in edges.values()
        )
        
        # Degree distribution
        degrees = [self.degree(node) for node in self.nodes()]
        
        return {
            "node_count": len(self._node_value),
            "edge_count": total_edges,
            "total_weight": total_weight,
            "average_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
            "density": total_edges / (len(self._node_value) * (len(self._node_value) - 1)) 
                      if len(self._node_value) > 1 else 0
        }

    # =======================
    #   STRING REPRESENTATION
    # =======================
    def __str__(self) -> str:
        stats = self.get_statistics()
        return (f"VersatileDigraph(nodes={stats['node_count']}, "
                f"edges={stats['edge_count']}, "
                f"avg_degree={stats['average_degree']:.2f})")

    def __repr__(self) -> str:
        return self.__str__()