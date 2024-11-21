import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import xml.etree.ElementTree as ET


#################################################

def show_single_route(nod_file_path: str, 
                      edg_file_path: str, 
                      path: list[str], 
                      origin: str, 
                      destination: str, 
                      **kwargs):
    # Parse the network
    nodes, edges = _parse_network_files(nod_file_path, edg_file_path)
    graph = _create_graph(nodes, edges)
    # Visualize the path
    visualize_path(graph, path, origin, destination, **kwargs)
    
#################################################


def visualize_path(graph: nx.DiGraph, path: list[str], origin_edge: str, destination_edge: str,
                   show: bool = True,
                   save_file_path: str | None = None,
                   title: str = "Path Visualization",
                   cmap_name: str = "Reds",
                   offset: float = 5.0,
                   figsize: tuple[int] = (12, 8),
                   xcrop: tuple[float, float] | None = None,
                   ycrop: tuple[float, float] | None = None) -> None:
    
    # Get node positions
    node_positions = nx.get_node_attributes(graph, 'pos')
    
    # Initiate the plot
    plt.figure(figsize=figsize)
    
    # Draw the full network
    nx.draw(graph, node_positions, node_size=10, node_color='lightblue', style='--', edge_color='gray', arrows=False)
    
    # Highlight OD edges
    origin_coords, dest_coords = None, None
    for source_node, target_node, edge_id in graph.edges(data=True):
        if edge_id['edge_id'] == origin_edge:
            origin_coords = (source_node, target_node)
        elif edge_id['edge_id'] == destination_edge:
            dest_coords = (source_node, target_node)
    try:
        nx.draw_networkx_edges(graph, node_positions, edgelist=[origin_coords, dest_coords], edge_color=["black", "black"], width=5)
    except:
        raise ValueError("Origin or destination edge not found in the graph.")
    
    # Draw the paths
    # Get the edge IDs and source-target nodes in the path
    path_edges_graph = {data_dict['edge_id']: (source, target) \
        for source, target, data_dict in graph.edges(data=True) if data_dict['edge_id'] in path}
    # Get colormap
    colors = _get_colors(len(path), cmap_name)
    # Draw the path edges one by one
    for edge_id, (source_node, target_node) in path_edges_graph.items():
        # Don't draw if it's origin or destination
        if edge_id in (origin_edge, destination_edge):    continue
        # Shift the edge by the offset
        new_pos = _shift_edge_by_offset(node_positions, source_node, target_node, offset)
        # Draw the edge
        color = colors[path.index(edge_id)]
        nx.draw_networkx_edges(graph, new_pos, edgelist=[(source_node, target_node)], edge_color=[color], width=3)
            
    # Crop the plot if requested
    if xcrop is not None:
        plt.xlim(xcrop)
    if ycrop is not None:
        plt.ylim(ycrop)
    
    # Set the title and show the plot
    plt.title(title) 
    fig = plt.gcf()   # Get the current figure
    fig.canvas.manager.set_window_title(title)
    
    # Save the plot if requested
    if save_file_path is not None:
        plt.savefig(save_file_path, bbox_inches='tight')
        
    if show:
        plt.show()
    

def _parse_network_files(nod_file, edg_file):
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


def _create_graph(nodes, edges):
    """
    Creates a directed graph from nodes and edges.
    """
    graph = nx.DiGraph()
    for node_id, coords in nodes.items():
        graph.add_node(node_id, pos=coords)
    for from_node, to_node, edge_id in edges:
        graph.add_edge(from_node, to_node, edge_id=edge_id)
    return graph
    

def _shift_edge_by_offset(node_positions, source, target, offset: float):
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
    
    
def _get_colors(num_colors: int, cmap_name: str):
    cmap = plt.get_cmap(cmap_name)
    cmap_truncated = mcolors.LinearSegmentedColormap.from_list("cmap_truncated", cmap(np.linspace(0.25, 1, 256)))
    norm = mcolors.Normalize(vmin=0, vmax=num_colors)
    colors = [cmap_truncated(norm(i)) for i in range(num_colors)]
    return colors
