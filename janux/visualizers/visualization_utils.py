import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import xml.etree.ElementTree as ET

def parse_network_files(nod_file, edg_file):
    """
    Parses nodes and edges from the given network files.
    """
    # Parse nodes
    node_tree = ET.parse(nod_file)
    nodes = {}
    for node in node_tree.findall("node"):
        node_id = node.get("id")
        x, y = float(node.get("x")), float(node.get("y"))
        nodes[node_id] = (x, y)
    # Parse edges
    edge_tree = ET.parse(edg_file)
    edges = []
    for edge in edge_tree.findall("edge"):
        edge_id = edge.get("id")
        from_node, to_node = edge.get("from"), edge.get("to")
        edges.append((from_node, to_node, edge_id))
    return nodes, edges


def create_graph(nodes, edges):
    """
    Creates a directed graph from nodes and edges.
    """
    graph = nx.DiGraph()
    for node_id, coords in nodes.items():
        graph.add_node(node_id, pos=coords)
    for from_node, to_node, edge_id in edges:
        graph.add_edge(from_node, to_node, edge_id=edge_id)
    return graph
    

def shift_edge_by_offset(node_positions, source, target, offset: float):
    # Get node positions
    x1, y1 = node_positions[source]
    x2, y2 = node_positions[target]
    # Compute perpendicular offset
    dx, dy = x2 - x1, y2 - y1
    length = (dx**2 + dy**2)**0.5
    offset_x = -dy / length * offset
    offset_y = dx / length * offset
    
    # Apply offset to edge positions
    new_pos = {
        source: (x1 + offset_x, y1 + offset_y),
        target: (x2 + offset_x, y2 + offset_y),
    }
    return new_pos
    
    
def get_colors(num_colors: int, cmap_name: str):
    cmap = plt.get_cmap(cmap_name)
    cmap_truncated = mcolors.LinearSegmentedColormap.from_list("cmap_truncated", cmap(np.linspace(0.25, 1, 256)))
    norm = mcolors.Normalize(vmin=0, vmax=num_colors)
    colors = [cmap_truncated(norm(i)) for i in range(num_colors)]
    return colors
