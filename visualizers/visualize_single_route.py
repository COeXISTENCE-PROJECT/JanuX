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
    Visualizes a single route on a transportation network.

    This function parses the network's node and edge files, constructs a graph 
    representation of the network, and visualizes a specific route along with 
    its origin and destination edges.

    Parameters:
    ----------
    nod_file_path : str
        Path to the network's node (.nod.xml) file. This file should define node IDs 
        and their coordinates (x, y).
    edg_file_path : str
        Path to the network's edge (.edg.xml) file. This file should define edge IDs 
        and their corresponding source and target nodes.
    path : list[str]
        A list of edge IDs representing the route to be visualized.
    origin : str
        The edge ID representing the starting point of the route.
    destination : str
        The edge ID representing the endpoint of the route.
    **kwargs : dict
        Additional keyword arguments to customize the visualization. These arguments 
        are passed directly to the `visualize_path` function.

    Keyword Arguments:
    ------------------
    show : bool, optional
        If True, the visualization will be displayed (default is True).
    save_file_path : str, optional
        Path to save the visualization image. If None, the image is not saved.
    title : str, optional
        Title of the visualization plot (default is "Path Visualization").
    cmap_name : str, optional
        Name of the colormap used to color the route (default is "Reds").
    offset : float, optional
        Vertical offset applied to the path edges for better visibility (default is 5.0).
    fig_width : int, optional
        Width of the figure in inches (default is 8).
    autocrop : bool, optional
        If True, the visualization is cropped to the route's extent (default is True). Overrides `xcrop` and `ycrop` if `True`.
    xcrop : tuple[float, float], optional
        Manual x-axis cropping range (default is None).
    ycrop : tuple[float, float], optional
        Manual y-axis cropping range (default is None).
    crop_margin : float, optional
        Margin added to the autocropping range (default is 10).

    Returns:
    -------
    None
        The function generates a visualization, which is displayed or saved based on 
        the provided keyword arguments.
    )
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
                   offset: float = 5.0,
                   fig_width: tuple[int] = 8,
                   autocrop: bool = True,
                   xcrop: tuple[float, float] | None = None,
                   ycrop: tuple[float, float] | None = None,
                   crop_margin: float = 10) -> None:
    
    # Get node positions
    node_positions = nx.get_node_attributes(graph, 'pos')
    
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
            nx.draw_networkx_edges(graph, new_pos, edgelist=[(source_node, target_node)], edge_color=[color], width=3)
        
        if autocrop:
            # Update the cropping limits
            x_max = max(x_max, new_pos[source_node][0], new_pos[target_node][0])
            x_min = min(x_min, new_pos[source_node][0], new_pos[target_node][0])
            y_max = max(y_max, new_pos[source_node][1], new_pos[target_node][1])
            y_min = min(y_min, new_pos[source_node][1], new_pos[target_node][1])
        
    # Crop the figure if requested    
    if autocrop:
        x_range = (x_min - crop_margin, x_max + crop_margin)
        y_range = (y_min - crop_margin, y_max + crop_margin)
        plt.xlim(x_range)
        plt.ylim(y_range)
        # Set figsize
        fig = plt.gcf()
        fig.set_size_inches(fig_width, fig_width * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0]))
    else:
        if xcrop is not None:
            plt.xlim(xcrop)
        if ycrop is not None:
            plt.ylim(ycrop)
        if xcrop is not None and ycrop is not None:
            # Set figsize
            fig = plt.gcf()
            fig.set_size_inches(fig_width, fig_width * (ycrop[1] - ycrop[0]) / (xcrop[1] - xcrop[0]))
    
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
