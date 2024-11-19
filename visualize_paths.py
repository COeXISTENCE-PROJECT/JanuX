import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import numpy as np


#################################################

def show_paths(nod_file_path, edg_file_path, paths, origin, destination, **kwargs):
    # Parse the network
    nodes, edges = parse_network_files(nod_file_path, edg_file_path)
    graph = create_graph(nodes, edges)
    # Visualize the path
    visualize_path(graph, paths, origin, destination, **kwargs)
    
#################################################


def visualize_path(graph, paths, origin_edge, destination_edge,
                   save_path=None,
                   title="Path Visualization",
                   cmap_names=['Reds', 'Blues', 'Greens', 'Purples', 'Oranges'],
                   offsets=[-6, -3, 3, 9],
                   figsize=(12, 8),
                   xcrop=None,
                   ycrop=None):
    """
    Visualizes paths on a directed graph with highlighted origin and destination edges.

    Parameters:
    -----------
    graph : networkx.DiGraph
        The directed graph containing nodes and edges with positional attributes.

    paths : list of list of str
        A list of paths, where each path is a list of edge IDs.

    origin_edge : str
        The edge ID representing the starting edge.

    destination_edge : str
        The edge ID representing the ending edge.

    title : str, optional
        The title of the plot. Default is "Path Visualization".

    cmap_names : list of str, optional
        A list of sequential colormap names to use for coloring the paths. Each path gets a unique colormap. 
        Default is ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges'].

    offsets : list of float, optional
        A list of offsets to separate the paths visually. Each path gets a unique offset. 
        Default is [-6, -3, 3, 9].

    figsize : tuple of float, optional
        The size of the figure (width, height) in inches. Default is (12, 8).

    xcrop : tuple of float, optional
        A tuple (xmin, xmax) to crop the x-axis of the plot. Default is None (no cropping).

    ycrop : tuple of float, optional
        A tuple (ymin, ymax) to crop the y-axis of the plot. Default is None (no cropping).

    Raises:
    -------
    AssertionError
        If the number of paths exceeds the number of available color maps or offsets.

    ValueError
        If the origin or destination edge is not found in the graph.

    Notes:
    ------
    - The function highlights the origin and destination edges in black.
    - Paths are visualized using unique colors for each edge along the path, excluding the origin and destination edges.
    - Paths are visually separated using the specified offsets.

    Example:
    --------
    visualize_path(graph, paths, 'edge_1', 'edge_2', title="My Path Plot")
    """
    
    assert len(paths) <= len(cmap_names), f"{len(cmap_names)} color maps is not variate enough to visualize {len(paths)} paths."
    assert len(paths) <= len(offsets), f"{len(offsets)} offsets is not variate enough to visualize {len(paths)} paths."
    
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
    for path_idx, path_edges in enumerate(paths):
        # Get the edge IDs and source-target nodes in the path
        path_edges_graph = {data_dict['edge_id']: (source, target) for source, target, data_dict in graph.edges(data=True) if data_dict['edge_id'] in path_edges}
        # Get colormap
        colors = _get_colors(len(path_edges), cmap_names[path_idx])
        # Draw the path edges one by one
        for edge_id, (source_node, target_node) in path_edges_graph.items():
            # Don't draw if it's origin or destination
            if edge_id in (origin_edge, destination_edge):    continue
            # Shift the edge by the offset
            new_pos = _shift_edge_by_offset(node_positions, source_node, target_node, offsets[path_idx])
            # Draw the edge
            color = colors[path_edges.index(edge_id)]
            nx.draw_networkx_edges(graph, new_pos, edgelist=[(source_node, target_node)], edge_color=[color], width=3)
            
    # Crop the plot if requested
    if xcrop is not None:
        plt.xlim(xcrop)
    if ycrop is not None:
        plt.ylim(ycrop)
    
    # Set the title and show the plot
    plt.title(title)
    
    # Save the plot if requested
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()
    

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
