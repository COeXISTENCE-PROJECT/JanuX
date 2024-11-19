import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import networkx as nx
import numpy as np
import pandas as pd

from keychain import Keychain as kc
from utils import get_params
from utils import list_to_string


#################################################

def basic_route_generator(network: nx.DiGraph, origins: list[str], destinations: list[str], **kwargs) -> pd.DataFrame:
    params = get_params("params.json")
    
    for key, value in kwargs.items():
        params[key] = value

    number_of_paths = params[kc.NUMBER_OF_PATHS]
    beta = params[kc.BETA]
    weight = params[kc.WEIGHT]
    num_samples = params[kc.NUM_SAMPLES]
    max_path_length = params[kc.MAX_PATH_LENGTH]

    origins = {i : origin for i, origin in enumerate(origins)}
    destinations = {i : dest for i, dest in enumerate(destinations)}

    check_od_integrity(network, origins, destinations)
    routes = create_routes(network, number_of_paths, origins, destinations, beta, weight, num_samples, max_path_length)
    routes = paths_to_df(routes, network, origins, destinations)
    return routes

#################################################


############## OD Integrity #################

def check_od_integrity(network, origins, destinations):
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

def create_routes(network: nx.DiGraph, num_routes: int, origins: list[str], destinations: list[str], \
    beta: float, weight: str, num_samples: int=50, max_path_length:int=100) -> dict:
    assert num_samples >= num_routes, f"Number of samples ({num_samples}) should be at least equal to the number of routes ({num_routes})"
    assert max_path_length > 0, f"Maximum path length should be greater than 0"
    routes = dict()   # Tuple<od_id, dest_id> : List<routes>
    for dest_idx, dest_name in destinations.items():
        node_potentials = dict(nx.shortest_path_length(network, target=dest_name, weight=weight))
        for origin_idx, origin_name in origins.items():
            sampled_routes = list()   # num_samples number of routes
            while (len(sampled_routes) < num_samples) or (len(set(sampled_routes)) < num_routes):
                path = _sample_single_route(network, origin_name, dest_name, node_potentials, beta, max_path_length)
                if not path is None:
                    sampled_routes.append(tuple(path))
                    print(f"\r[INFO] Sampled {len(sampled_routes)} paths for {origin_idx} -> {dest_idx}", end="")
            routes[(origin_idx, dest_idx)] = _pick_routes_from_samples(sampled_routes, num_routes)
            print(f"\n[INFO] Selected {len(set(routes[(origin_idx, dest_idx)]))} paths for {origin_idx} -> {dest_idx}")
    return routes


def _pick_routes_from_samples(sampled_routes: list[tuple], num_paths: int):
    assert num_paths <= len(sampled_routes), f"Number of paths ({num_paths}) should be less than or equal to the number of sampled routes ({len(sampled_routes)})"
    assert num_paths > 0, f"Number of paths should be greater than 0"
    sampled_routes = np.array(sampled_routes, dtype=object)
    # Get each unique route and their counts
    unique_routes, route_counts = np.unique(sampled_routes, return_counts=True)
    # Calculate sampling probabilities (according to their counts)
    sampling_probabilities = route_counts / route_counts.sum()
    # Sample from the unique items according to the probabilities
    assert num_paths <= len(unique_routes), f"Cannot sample {num_paths} distinct items from {len(unique_routes)} unique items."
    picked_routes = np.random.choice(unique_routes, size=num_paths, p=sampling_probabilities, replace=False)
    return picked_routes


def _sample_single_route(network, origin, destination, node_potentials, beta, maxlen):
    path, current_node = list(), origin
    while True:
        path.append(current_node)
        options = [node for node in network.neighbors(current_node) if (node not in path)]
        if   (destination in options):                  return path + [destination]
        elif (not options) or (len(path) > maxlen):     return None
        else:       
            try:            
                current_node = _logit(options, node_potentials, beta)
            except:
                return None


def _logit(options, node_potentials: dict, beta):
    # If a node does not have a potential, it is a dead end, so we assign an infinite potential
    numerators = [np.exp(beta * node_potentials.get(option, float("inf"))) for option in options]
    utilities = [numerator/sum(numerators) for numerator in numerators]
    return np.random.choice(options, p=utilities)

#################################################


################## FF Times #####################


def _calculate_free_flow_time(route: list[str], network: nx.DiGraph) -> float:
    """
    Calculate the free-flow travel time for a given route based on the network edges.

    Args:
        route (list[str]): A list of nodes representing the route.
        network (nx.DiGraph): The directed graph representing the traffic network.

    Returns:
        float: The total free-flow travel time for the route.
    """
    # Create a DataFrame with edge attributes from the network
    edges_df = pd.DataFrame(network.edges(data=True), columns=["source", "target", "attributes"])

    # Extract travel time from edge attributes and clean up its format
    edges_df["travel_time"] = (
        edges_df["attributes"].astype('str').str.split(':',expand=True)[1].replace('}','',regex=True).astype('float')
    )
    
    # Initialize total travel time
    total_travel_time = 0.0

    # Iterate through consecutive nodes in the route to calculate travel time
    for source, target in zip(route[:-1], route[1:]):
        # Filter for the matching edge in the DataFrame
        matching_edge = edges_df[(edges_df["source"] == source) & (edges_df["target"] == target)]

        if not matching_edge.empty:
            total_travel_time += matching_edge["travel_time"].iloc[0]
        else:
            raise ValueError(f"No edge found between {source} and {target} in the network.")

    return total_travel_time

#################################################


################## To DF #####################

def paths_to_df(routes: dict, network: nx.DiGraph, origins: dict, destinations: dict) -> pd.DataFrame:
    """
    Create a DataFrame from the routes with origin, destination, path, and free-flow time.

    Args:
        routes (dict): A dictionary where keys are (origin_idx, dest_idx) and values are lists of routes.
        network (nx.DiGraph): The directed graph representing the traffic network.
        origins (dict): A dictionary mapping origin indices to origin names.
        destinations (dict): A dictionary mapping destination indices to destination names.

    Returns:
        pd.DataFrame: A DataFrame containing the origin, destination, path (as a string), and free-flow time.
    """
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
            free_flow = _calculate_free_flow_time(path, network)
            # Append the row to the DataFrame
            paths_df.loc[len(paths_df.index)] = [origin_name, dest_name, path_as_str, free_flow]
    return paths_df

#################################################