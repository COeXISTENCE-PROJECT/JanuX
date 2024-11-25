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
    """
    Displays a visualization of a single route on a graph derived from given node and edge files.

    Parameters:
    -----------
    nod_file_path : str
        Path to the node file containing node definitions in XML format.
    edg_file_path : str
        Path to the edge file containing edge definitions in XML format.
    path : list[str]
        A list of edge IDs that represent the route to be visualized.
    origin : str
        The edge ID for the origin of the route.
    destination : str
        The edge ID for the destination of the route.
    **kwargs : dict
        Additional keyword arguments passed to `visualize_path` for customization.

    Keyword Arguments (`kwargs`):
    -----------------------------
    show : bool, default=True
        Whether to display the plot.
    save_file_path : str | None, default=None
        File path to save the generated plot as an image. If None, the plot is not saved.
    title : str, default="Path Visualization"
        Title of the plot window and the saved figure.
    cmap_name : str, default="Reds"
        Name of the colormap used for coloring the path (sequential).
    offset : float, default=0
        Shift to be applied to every edge, can be used to not obstruct the original edge.
    fig_size : tuple[int], default=(12, 8)
        Size of the figure (width, height) in inches.
    autocrop : bool, default=True
        Whether to crop the figure to focus on the path and reduce unnecessary whitespace.
    autocrop_margin : float, default=10
        Margin in data units to add around the autocropped view.
    xcrop : tuple[float, float] | None, default=None
        Fixed x-axis range for the plot. Overriden by `autocrop`.
    ycrop : tuple[float, float] | None, default=None
        Fixed y-axis range for the plot. Overriden by `autocrop`.
    node_size : int, default=10
        Size of the nodes in the plot.
    node_color : str, default='lightblue'
        Color of the nodes in the plot.
    path_width : int, default=5
        Width of the edges that represent the path in the plot.

    Returns:
    --------
    None
        Displays the visualization of the route and optionally saves it to a file.
    """
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
                   offset: float = 0,
                   fig_size: tuple[int] = (12, 8),
                   autocrop: bool = True,
                   autocrop_margin: float = 10,
                   xcrop: tuple[float, float] | None = None,
                   ycrop: tuple[float, float] | None = None,
                   node_size: int = 10,
                   node_color: str = 'lightblue',
                   path_width: int = 5) -> None:
    
    # Initilize the figure
    plt.figure(figsize=fig_size)
    
    # Get node positions
    node_positions = nx.get_node_attributes(graph, 'pos')
    
    # Draw the full network
    nx.draw(graph, node_positions, node_size=node_size, node_color=node_color, style='--', edge_color='gray', arrows=False)
    
    # Highlight OD edges
    origin_coords, dest_coords = None, None
    for source_node, target_node, edge_id in graph.edges(data=True):
        if edge_id['edge_id'] == origin_edge:
            origin_coords = (source_node, target_node)
        elif edge_id['edge_id'] == destination_edge:
            dest_coords = (source_node, target_node)
    try:
        nx.draw_networkx_edges(graph, node_positions, edgelist=[origin_coords, dest_coords], edge_color=["black", "black"], width=path_width)
    except:
        raise ValueError("Origin or destination edge not found in the graph.")
    
    x_max, x_min, y_max, y_min = float('-inf'), float('inf'), float('-inf'), float('inf')
    
    # Draw the paths
    # Get the edge IDs and source-target nodes in the path
    path_edges_graph = {data_dict['edge_id']: (source, target) \
        for source, target, data_dict in graph.edges(data=True) if data_dict['edge_id'] in path}
    # Get colormap
    colors = _get_colors(len(path), cmap_name)
    # Draw the path edges one by one
    for edge_id, (source_node, target_node) in path_edges_graph.items():
        # Shift the edge by the offset
        new_pos = _shift_edge_by_offset(node_positions, source_node, target_node, offset)
        # Draw the edge
        color = colors[path.index(edge_id)]
        # Draw if it's not origin or destination
        if edge_id not in (origin_edge, destination_edge):
            nx.draw_networkx_edges(graph, new_pos, edgelist=[(source_node, target_node)], edge_color=[color], width=path_width)
        
        if autocrop:
            # Update the cropping limits
            x_max = max(x_max, new_pos[source_node][0], new_pos[target_node][0])
            x_min = min(x_min, new_pos[source_node][0], new_pos[target_node][0])
            y_max = max(y_max, new_pos[source_node][1], new_pos[target_node][1])
            y_min = min(y_min, new_pos[source_node][1], new_pos[target_node][1])
        
# Crop the figure if requested
    if autocrop:
        # Determine the aspect ratio of the ranges
        x_range_length = x_max - x_min
        y_range_length = y_max - y_min
        cropped_aspect_ratio = y_range_length / x_range_length
        
        fig_width, fig_height = fig_size  # Base figure size
        fig_aspect_ratio = fig_height / fig_width

        if cropped_aspect_ratio > fig_aspect_ratio:
            # Cropped aspect ratio is taller than the base aspect ratio
            x_range_length_new = (cropped_aspect_ratio / fig_aspect_ratio) * x_range_length
            difference = x_range_length_new - x_range_length
            # Expand the image on x axis
            x_max += difference / 2
            x_min -= difference / 2
        else:
            # Cropped aspect ratio is wider than the base aspect ratio
            y_range_length_new = (fig_aspect_ratio / cropped_aspect_ratio) * y_range_length
            difference = y_range_length_new - y_range_length
            # Expand the image on y axis
            y_max += difference / 2
            y_min -= difference / 2

        # Set plot limits
        plt.xlim(x_min-autocrop_margin, x_max+autocrop_margin)
        plt.ylim(y_min-autocrop_margin, y_max+autocrop_margin)
    else:    
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
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        
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
