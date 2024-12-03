import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from typing import Union

from janux.visualizers.visualization_utils import create_graph
from janux.visualizers.visualization_utils import parse_network_files


#################################################

def show_edge_attributes(nod_file_path: str,
                                  edg_file_path: str,
                                  congestion_dict: dict,
                                  **kwargs):
    """
    Displays a visualization of congestion levels on a graph derived from given node and edge files.

    Parameters:
    -----------
    nod_file_path : str
        Path to the node file containing node definitions in XML format.
    edg_file_path : str
        Path to the edge file containing edge definitions in XML format.
    congestion_dict : dict
        A dictionary where keys are edge IDs and values are numerical congestion levels.
    **kwargs : dict
        Additional keyword arguments passed to `visualize_congestion` for customization.

    Keyword Arguments (`kwargs`):
    -----------------------------
    show : bool, default=True
        Whether to display the plot.
    save_file_path : str | None, default=None
        File path to save the generated plot as an image. If None, the plot is not saved.
    title : str, default="Congestion Visualization"
        Title of the plot window and the saved figure.
    cmap_name : str, default="Reds"
        Name of the colormap used for coloring the edges.
    fig_size : tuple[int], default=(12, 8)
        Size of the figure (width, height) in inches.

    Returns:
    --------
    None
        Displays the visualization of the congestion levels and optionally saves it to a file.
    """
    # Parse the network
    nodes, edges = parse_network_files(nod_file_path, edg_file_path)
    graph = create_graph(nodes, edges)
    # Visualize congestion
    visualize_congestion(graph, congestion_dict, **kwargs)


#################################################

def visualize_congestion(graph: nx.DiGraph, congestion_dict: dict,
                         show: bool = True,
                         save_file_path: Union[str, None] = None,
                         title: str = "Congestion Visualization",
                         cmap_name: str = "Reds",
                         autocrop: bool = True,
                         fig_size: tuple[int] = (12, 8)) -> None:
    
    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Get node positions
    node_positions = nx.get_node_attributes(graph, 'pos')
    
    # Draw the full network with default style
    nx.draw(graph, node_positions, node_size=10, node_color='lightblue', style='--', edge_color='gray', arrows=False, ax=ax)
    
    # Get colormap
    cmap = plt.get_cmap(cmap_name)
    cmap = mcolors.LinearSegmentedColormap.from_list("cmap_truncated", cmap(np.linspace(0.25, 1, 256)))
    norm = mcolors.Normalize(vmin=min(congestion_dict.values()), vmax=max(congestion_dict.values()))
    
    x_max, x_min, y_max, y_min = float('-inf'), float('inf'), float('-inf'), float('inf')
    
    # Draw edges with congestion-based coloring
    edge_colors = []
    edge_widths = []
    edges = []
    for source_node, target_node, edge_id in graph.edges(data=True):
        if edge_id['edge_id'] in congestion_dict and autocrop:
            # Update the cropping limits
            x_max = max(x_max, node_positions[source_node][0], node_positions[target_node][0])
            x_min = min(x_min, node_positions[source_node][0], node_positions[target_node][0])
            y_max = max(y_max, node_positions[source_node][1], node_positions[target_node][1])
            y_min = min(y_min, node_positions[source_node][1], node_positions[target_node][1])
        congestion_value = congestion_dict.get(edge_id['edge_id'], 0)  # Default to 0 if not in dict
        if edge_id['edge_id'] in congestion_dict:
            color = cmap(norm(congestion_value))
        else:
            # transparent color
            color = (0, 0, 0, 0)
        edge_colors.append(color)
        edge_widths.append(3 + (congestion_value))  # Adjust width based on congestion level
        edges.append((source_node, target_node))

    edge_collection = nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=edges,
        edge_color=edge_colors,
        width=edge_widths,
        ax=ax,
        arrows=False
    )
    
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
        plt.xlim(x_min-10, x_max+10)
        plt.ylim(y_min-10, y_max+10)
    
    # Set the title and show the plot
    ax.set_title(title)
    
    # Save the plot if requested
    if save_file_path is not None:
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        
    if show:
        fig.canvas.manager.set_window_title(title)
        plt.show()
        
    plt.close()