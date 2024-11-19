import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import inspect
import networkx as nx
import numpy as np
import pandas as pd

from typing import Callable

from keychain import Keychain as kc
from path_generators import calculate_free_flow_time
from utils import get_params
from utils import list_to_string


#################################################

def heuristic_path_generator(network: nx.DiGraph, 
                          origins: list[str], 
                          destinations: list[str], 
                          heuristics: list[Callable],
                          heur_weights: list[float],
                          **kwargs) -> pd.DataFrame:
            
    """
    Generate heuristic-based paths between origin-destination pairs in a directed graph.

    This function computes a set of routes between specified origins and destinations 
    in the network using user-defined heuristic functions and weights. The generated 
    paths are returned in a DataFrame, including their free-flow travel times.

    Parameters:
    ----------
    network : nx.DiGraph
        A directed graph representing the network. Nodes are locations, and edges 
        include attributes such as weights for travel time.
    origins : list[str]
        A list of origin node names.
    destinations : list[str]
        A list of destination node names.
    heuristics : list[Callable]
        A list of heuristic functions used to score and rank routes. Each function 
        should accept paths and the network as arguments and return a numerical score.
    heur_weights : list[float]
        A list of weights corresponding to the heuristic functions, defining their 
        relative importance in route selection.
    **kwargs : dict, optional
        Additional parameters to update default settings from the `params.json` file. 
        These may include:
        - `number_of_paths`: Number of paths to generate for each origin-destination pair.
        - `beta`: Parameter controlling the influence of node potentials in path sampling.
        - `weight`: Edge attribute to use as weight in the network (e.g., time).
        - `num_samples`: Number of route samples for evaluation.
        - `max_path_length`: Maximum allowable length of a path in terms of the number of nodes.
        - `random_seed`: Seed for reproducible random number generation.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the generated paths with the following columns:
        - `origins`: Origin node name.
        - `destinations`: Destination node name.
        - `path`: The route as a string representation of node names.
        - `free_flow_time`: Free-flow travel time for the route.

    Raises:
    ------
    AssertionError
        - If the origins or destinations are not in the network or not connected.
        - If invalid parameters are provided.

    Notes:
    -----
    - The function ensures reproducibility when a random seed is provided.
    - Paths are evaluated using a weighted combination of user-defined heuristic functions, 
      and the top-ranked paths are selected.
    - Routes are generated using a probabilistic logit-based approach guided by 
      node potentials.
    """        
    
    # Get parameters from the params.json file and update them with the provided kwargs
    params = get_params("params.json")
    params.update(kwargs)

    # Get parameters
    number_of_paths = params[kc.NUMBER_OF_PATHS]
    beta = params[kc.BETA]
    weight = params[kc.WEIGHT]
    num_samples = params[kc.NUM_SAMPLES]
    max_path_length = params[kc.MAX_PATH_LENGTH]
    
    # Set random seed if provided
    random_seed = params.get(kc.RANDOM_SEED, None)
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    # Convert origin and destination names to indices
    origins = {i : origin for i, origin in enumerate(origins)}
    destinations = {i : dest for i, dest in enumerate(destinations)}

    # Check if the origins and destinations are in the network and connected
    check_od_integrity(network, origins, destinations)
    # Generate routes
    routes = create_routes(network, number_of_paths, origins, destinations, heuristics,
                           heur_weights, beta, weight, rng, num_samples, max_path_length)
    # Convert routes to a DataFrame
    routes_df = paths_to_df(routes, network, origins, destinations)
    return routes_df

#################################################


############## OD Integrity #################

def check_od_integrity(network: nx.DiGraph, origins: list[str], destinations: list[str]) -> None:
    for origin in origins.values():
        assert origin in network.nodes, f"Origin {origin} is not in the network"
    for destination in destinations.values():
        assert destination in network.nodes, f"Destination {destination} is not in the network"
        
    for origin_idx, origin in origins.items():
        paths_from_origin = nx.multi_source_dijkstra_path(network, sources=[origin])
        for dest_idx, destination in destinations.items():
            assert destination in paths_from_origin, f"Origin {origin_idx} cannot reach destination {dest_idx}"

#################################################


############## Route Generation #################

def create_routes(network: nx.DiGraph, num_routes: int, origins: list[str], destinations: list[str],
                  heuristics: list[Callable], heur_weights: list[float], beta: float, weight: str, 
                  rng: np.random.Generator, num_samples: int=50, max_path_length:int=100) -> dict:
    
    """
    Generates a set of routes between origin-destination pairs in a directed graph using heuristic-based sampling.

    This function samples multiple routes between specified origin and destination nodes in the graph and selects 
    the most suitable routes based on user-defined heuristics and their weights. It leverages probabilistic sampling 
    with logit-based decision-making and evaluates the routes with weighted user-defined heuristic scores.

    Parameters:
    ----------
    network : nx.DiGraph
        A directed graph representing the network. Nodes represent locations, and edges include weights such as travel time.
    num_routes : int
        The number of routes to select for each origin-destination pair.
    origins : list[str]
        A dictionary mapping origin indices to their corresponding node names.
    destinations : list[str]
        A dictionary mapping destination indices to their corresponding node names.
    heuristics : list[Callable]
        A list of heuristic functions used to score and rank sampled routes. Each function should accept 
        paths and the network as arguments and return a numerical score.
    heur_weights : list[float]
        A list of weights corresponding to each heuristic function, defining their relative importance.
    beta : float
        A parameter controlling the influence of node potentials during route sampling. Higher values 
        make the sampling more deterministic.
    weight : str
        The edge attribute in the graph used as the weight for calculations (e.g., 'travel_time').
    rng : np.random.Generator
        A random number generator for reproducibility during sampling.
    num_samples : int, optional
        The number of route groups to sample for evaluation, by default 50.
    max_path_length : int, optional
        The maximum allowable length of a path in terms of the number of nodes, by default 100.

    Returns:
    -------
    dict
        A dictionary where keys are tuples of (origin_index, destination_index), and values are lists of selected routes. 
        Each route is represented as a tuple of node names.

    Raises:
    ------
    AssertionError
        - If `num_samples` is less than `num_routes`.
        - If `max_path_length` is not positive.

    Notes:
    -----
    - Routes are generated using a probabilistic sampling approach guided by node potentials and a logit function.
    - The routes are scored using a weighted combination of heuristic functions, and the top `num_routes` are selected.
    - The function ensures no duplicate routes are added to the selected set, which is different from `basic_path_generator`.

    """
    
    assert num_samples >= num_routes, f"Number of samples ({num_samples}) should be at least equal to the number of routes ({num_routes})"
    assert max_path_length > 0, f"Maximum path length should be greater than 0"
    
    routes = dict()   # Tuple<od_id, dest_id> : List<routes>
    for dest_idx, dest_name in destinations.items():
        node_potentials = dict(nx.shortest_path_length(network, target=dest_name, weight=weight))
        for origin_idx, origin_name in origins.items():
            sampled_routes = set()   # num_samples number of routes
            while (len(sampled_routes) < num_samples):
                path = _sample_single_route(network, origin_name, dest_name, node_potentials, beta, max_path_length, rng)
                if not path is None:
                    sampled_routes.add(tuple(path))
                    print(f"\r[INFO] Sampled {len(sampled_routes)} paths for {origin_idx} -> {dest_idx}", end="")
            sampled_routes = sorted(list(sampled_routes), key=lambda x: list_to_string(x))
            routes[(origin_idx, dest_idx)] = _pick_routes_from_samples(network, sampled_routes, num_routes, num_samples, 
                                                                       heuristics, heur_weights, rng)
            print(f"\n[INFO] Selected {len(set(routes[(origin_idx, dest_idx)]))} paths for {origin_idx} -> {dest_idx}")
    return routes


def _pick_routes_from_samples(network: nx.DiGraph, sampled_routes: list[tuple], num_paths: int, num_samples: int, 
                              heuristics: list[Callable], heur_weights: list[float], rng: np.random.Generator) -> list[tuple]:

    """
    Selects a set of optimal routes from sampled routes based on user-defined heuristic scores.

    This function evaluates groups of sampled routes using provided heuristics and their associated weights. 
    It then selects the group of routes with the highest group heuristic score.

    Parameters:
    ----------
    network : nx.DiGraph
        A directed graph representing the network where nodes are locations and edges contain weights (e.g., travel time).
    sampled_routes : list[tuple]
        A list of sampled routes, where each route is represented as a tuple of node names.
    num_paths : int
        The number of paths to select for each origin-destination pair.
    num_samples : int
        The number of route groups to sample for evaluation.
    heuristics : list[Callable]
        A list of heuristic functions. Each function calculates a desirability score based on a group of paths and the network.
        Each heuristic function must accept arguments (path1, ..., pathN, network), in total `num_paths+1` arguments. 
        Each heuristic function must return a numerical score, deterministically evaluating the desirability of the group of paths.
        Scores from heuristic functions are combined using the provided heuristic weights, to be used in the ranking of all group samples.
    heur_weights : list[float]
        A list of weights corresponding to each heuristic function, defining their relative importance.
    rng : np.random.Generator
        A random number generator for reproducibility during sampling.

    Returns:
    -------
    list[tuple]
        A list of selected routes, where each route is a tuple of node names.

    Raises:
    ------
    AssertionError
        - If `num_paths` is not greater than 0 or exceeds the number of `sampled_routes`.
        - If the number of heuristics and the number of heuristic weights do not match.
        - If the heuristic functions do not meet the expected argument and return type requirements.

    Notes:
    -----
    - The function evaluates route groups by sampling combinations of `num_paths` routes from the `sampled_routes`.
    - The scores for each route group are computed as a weighted sum of the heuristic values.
    - Only deterministic heuristics with consistent results are supported.
    - The group of routes (size of `num_paths`) with the highest group score is selected as the output.
    """

    assert num_paths <= len(sampled_routes), f"Number of paths ({num_paths}) should be less than or equal to the number of sampled routes ({len(sampled_routes)})"
    assert num_paths > 0, f"Number of paths should be greater than 0"
    _validate_heuristics(heuristics, heur_weights, num_paths, network, sampled_routes)
    
    # Sample groups of routes (as their indices in the sampled_routes list)
    route_group_samples = set()
    while len(route_group_samples) < num_samples:
        sampled_group = rng.choice(len(sampled_routes), size=num_paths, replace=False)
        sampled_group = tuple(sorted(sampled_group))
        route_group_samples.add(sampled_group)
    route_group_samples = list(route_group_samples)
    
    # Calculate scores for each group of sampled paths
    scores = list()
    for sampled_group in route_group_samples:
        score = 0
        paths = [sampled_routes[i] for i in sampled_group]
        for heur_weight, heuristic in zip(heur_weights, heuristics):
            score += heur_weight * heuristic(*paths, network)
        scores.append(score)
    
    # Select the group of sampled paths with the highest score
    selected_paths = route_group_samples[np.argmax(scores)]
    selected_paths = [sampled_routes[i] for i in selected_paths]
    return selected_paths


def _sample_single_route(network: nx.DiGraph, origin: str, destination: str, node_potentials: dict,
                         beta: float, maxlen: int, rng: np.random.Generator) -> list[str] | None:
    """
    Sample a single route between an origin and a destination in a directed graph.

    This function attempts to generate a route from the `origin` to the `destination` in the `network` 
    using a logit-based probabilistic model for node selection. The function considers node potentials 
    to guide the sampling process and respects a maximum path length constraint.

    Parameters:
    ----------
    network : nx.DiGraph
        A directed graph representing the network. Nodes are locations, and edges have attributes such 
        as weights or travel times.
    origin : str
        The starting node for the route.
    destination : str
        The target node for the route.
    node_potentials : dict
        A dictionary mapping node names to their potential values, used for probabilistic sampling.
    beta : float
        A parameter controlling the influence of node potentials on the sampling process. Higher values 
        make the sampling more deterministic.
    maxlen : int
        The maximum allowable length of the path in terms of the number of nodes.
    rng : np.random.Generator
        A random number generator for reproducibility.

    Returns:
    -------
    list[str] | None
        - A list of node names representing the sampled path, if a valid route is found.
        - `None` if the destination cannot be reached or the path exceeds the maximum length.

    Notes:
    ------
    - The function ensures that no node is revisited during the path generation process.
    - If no valid route is found within the constraints, `None` is returned.
    """
    path, current_node = list(), origin
    while True:
        path.append(current_node)
        options = [node for node in sorted(network.neighbors(current_node)) if node not in path]
        if   (destination in options):                  return path + [destination]
        elif (not options) or (len(path) > maxlen):     return None
        else:       
            try:            
                current_node = _logit(options, node_potentials, beta, rng)
            except:
                return None


def _logit(options: list, node_potentials: dict, beta: float, rng: np.random.Generator) -> str:
    # If a node does not have a potential, it is a dead end, so we assign an infinite potential
    numerators = [np.exp(beta * node_potentials.get(option, float("inf"))) for option in options]
    utilities = [numerator/sum(numerators) for numerator in numerators]
    return rng.choice(options, p=utilities)

#################################################


################## To DF #####################

def paths_to_df(routes: dict, network: nx.DiGraph, origins: dict, destinations: dict) -> pd.DataFrame:
    # Initialize an empty DataFrame with the required columns
    columns = [kc.ORIGINS, kc.DESTINATIONS, kc.PATH, kc.FREE_FLOW_TIME]
    paths_df = pd.DataFrame(columns=columns)
    # Iterate through the routes dictionary
    for (origin_idx, dest_idx), paths in routes.items():
        # Retrieve node names of the OD
        origin_name = origins[origin_idx]
        dest_name = destinations[dest_idx]
        for path in paths:
            # Convert the path to a string format
            path_as_str = list_to_string(path, ",")
            # Calculate the free-flow travel time for the path
            free_flow = calculate_free_flow_time(path, network)
            # Append the row to the DataFrame
            paths_df.loc[len(paths_df.index)] = [origin_name, dest_name, path_as_str, free_flow]
    return paths_df

#################################################

################## Helpers #####################

def _validate_heuristics(heuristics: list[Callable], heur_weights: list[float], num_paths: int, network: nx.DiGraph, sampled_routes: list[tuple]) -> None:
    # Assert that each heuristic accepts as argument number_of_paths+1 items (paths and network)
    for heuristic in heuristics:
        # Check if the heuristic is callable
        assert callable(heuristic), f"Each heuristic must be callable, but found: {type(heuristic)}"
        # Check the number of arguments
        num_arguments = len(inspect.signature(heuristic).parameters)
        assert num_arguments == num_paths+1, f"Each heuristic must accept exactly {num_paths+1} arguments, but {heuristic.__name__} takes {num_arguments}."
        # Check if return is numerical
        assert isinstance(heuristic(*sampled_routes[:num_paths], network), (int, float)), f"Each heuristic must return a numerical value, but {heuristic.__name__} returns {type(heuristic(*[[] for i in range(num_paths)], network))}"
        # Check if the heuristic is deterministic
        assert heuristic(*sampled_routes[:num_paths], network) == heuristic(*sampled_routes[:num_paths], network), f"Each heuristic must be deterministic, but {heuristic.__name__} is not."
    assert len(heur_weights) == len(heuristics), f"Number of heuristic weights does not match with number of heuristics. ({len(heur_weights)} and {len(heuristics)})"
        
#################################################