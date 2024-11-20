import networkx as nx
import pandas as pd

from typing import Callable

########################## Basic Path Generator ##########################

from .basic_generator import BasicPathGenerator

def basic_generator(network: nx.DiGraph,
                    origins: list[str], 
                    destinations: list[str], 
                    **kwargs) -> pd.DataFrame:
    """
    Generates routes for a given network using the BasicPathGenerator.

    This function initializes a BasicPathGenerator with the provided network, origins,
    and destinations, along with additional optional parameters. It generates routes
    for all specified origin-destination pairs and returns them in a structured DataFrame.
    For more information on the BasicPathGenerator, refer to the class documentation.

    Args:
        network (nx.DiGraph): The directed graph representing the network.
        origins (list[str]): A list of origin node names in the network.
        destinations (list[str]): A list of destination node names in the network.
        **kwargs: Additional arguments to customize the behavior of the BasicPathGenerator.

    Returns:
        pd.DataFrame: A DataFrame containing the generated routes with the following columns:
            - `origins`: The origin node for each route.
            - `destinations`: The destination node for each route.
            - `path`: A string representation of the nodes in the route.
            - `free_flow_time`: The travel time for the route under free-flow conditions.
    """
    generator = BasicPathGenerator(network, origins, destinations, **kwargs)
    return generator.generate_routes()

###########################################################################

######################### Heuristic Path Generator ########################

from .heuristic_based_generator import HeuristicPathGenerator

def heuristic_generator(network: nx.DiGraph, 
                              origins: list[str], 
                              destinations: list[str], 
                              heuristics: list[Callable], 
                              heur_weights: list[float], 
                              **kwargs) -> pd.DataFrame:
    """
    Generates routes for a given network using the HeuristicPathGenerator.

    This function initializes a HeuristicPathGenerator with the provided network, origins,
    destinations, heuristics, and their respective weights. The generator uses heuristic-based
    scoring to select the most optimal paths for each origin-destination pair. Additional
    parameters can be provided to customize the path generation process.
    For more information on the HeuristicPathGenerator, refer to the class documentation.

    Args:
        network (nx.DiGraph): The directed graph representing the network.
        origins (list[str]): A list of origin node names in the network.
        destinations (list[str]): A list of destination node names in the network.
        heuristics (list[Callable]): A list of heuristic functions used to evaluate route groups.
            Each function should be callable and accept the generated paths and network as inputs.
        heur_weights (list[float]): A list of weights corresponding to each heuristic, defining
            the influence of each heuristic on the overall score.
        **kwargs: Additional arguments to customize the behavior of the HeuristicPathGenerator.

    Returns:
        pd.DataFrame: A DataFrame containing the generated routes with the following columns:
            - `origins`: The origin node for each route.
            - `destinations`: The destination node for each route.
            - `path`: A string representation of the nodes in the route.
            - `free_flow_time`: The travel time for the route under free-flow conditions.
    """
    generator = HeuristicPathGenerator(network, origins, destinations, heuristics, heur_weights, **kwargs)
    return generator.generate_routes()

###########################################################################

