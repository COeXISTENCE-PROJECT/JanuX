import lxml
import networkx as nx
import numpy as np
import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

from keychain import Keychain as kc
from utils import df_to_prettytable
from utils import get_params
from utils import list_to_string


############## OD Integrity #################

def check_od_integrity(network, origins, destinations):
    # RK: @Onur TO DO this can be replaced with: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.multi_source_dijkstra_path.html#networkx.algorithms.shortest_paths.weighted.multi_source_dijkstra_path
    for dest_idx, destination in destinations.items():
        if not destination in network.nodes:    raise ValueError(f"Destination {dest_idx} is not in the network")
        distances_to_destination = dict(nx.shortest_path_length(network, target=destination))
        for origin_idx, origin in origins.items():
            if not origin in network.nodes:     raise ValueError(f"Origin {origin_idx} is not in the network")
            elif not origin in distances_to_destination:
                raise ValueError(f"Origin {origin_idx} cannot reach destination {dest_idx}")

#################################################


############## Route Generation #################

def create_routes(network, num_routes, origins, destinations, beta, weight, coeffs, num_samples=50, max_path_length=100):
    routes = dict()   # Tuple<od_id, dest_id> : List<routes>
    for dest_idx, dest_code in destinations.items():
        proximity_func = _get_proximity_function(network, dest_code, weight)   # Maps node -> proximity (cost)
        for origin_idx, origin_code in origins.items():
            sampled_routes = set()   # num_samples number of routes
            while len(sampled_routes) < num_samples:
                path = _path_generator(network, origin_code, dest_code, proximity_func, beta, max_path_length)
                if not path is None:
                    sampled_routes.add(tuple(path))
                    print(f"\r[INFO] Sampled {len(sampled_routes)} paths for {origin_idx} -> {dest_idx}", end="")
            routes[(origin_idx, dest_idx)] = _pick_routes_from_samples(sampled_routes, proximity_func, num_routes, coeffs, network)
            print(f"\n[INFO] Selected {len(routes[(origin_idx, dest_idx)])} paths for {origin_idx} -> {dest_idx}")
    return routes


def _get_proximity_function(network, destination, weight):
    # cost for all nodes that CAN access to destination
    # RK: This is typically called node potential
    distances_to_destination = dict(nx.shortest_path_length(network, target=destination, weight=weight))
    # dead-end nodes have infinite cost
    dead_nodes = [node for node in network.nodes if node not in distances_to_destination]
    for node in dead_nodes:  distances_to_destination[node] = float("inf")
    # return the lambda function
    return lambda x: distances_to_destination[x]


def _pick_routes_from_samples(sampled_routes, proximity, num_paths, coeffs, network):
    # RK: this function selects _num_paths_ routes of minimal utilty from the sampled
    # what this should do is to pick them randomly (or with some non-uniform probability)
    sampled_routes = list(sampled_routes)
    # what we base our selection on
    utility_dist = _get_route_utilities(sampled_routes, proximity, coeffs, network)
    # route indices that maximize defined utilities
    sorted_indices = np.argsort(utility_dist)[::-1] #RK: I'd say this can be sampled with prob of utility and not simply "pick n best"
    paths_idcs = sorted_indices[:num_paths]
    
    return [sampled_routes[idx] for idx in paths_idcs]


def _get_route_utilities(sampled_routes, proximity_func, coeffs, network):
    # RK: I would rename it to heuristics. And stick utility to something usual, namely a linear combination of distance, cost and time.

    # Based on FF times
    free_flows = [calculate_free_flow_time(route, network) for route in sampled_routes]
    utility1 = 1 / np.array(free_flows)
    utility1 = utility1 / np.sum(utility1) # RK: what is this formula? why not simply a sum of free flows?

    # Based on number of edges
    route_lengths = [len(route) for route in sampled_routes]
    utility2 = 1 / np.array(route_lengths)
    utility2 = utility2 / np.sum(utility2) # this is never called length in transportation. In transport we use physical length and graph-theoretical length (number of hops) is almost never used as very sensitive to modelling artifacts.

    # Based on proximity increase in consecutive nodes (how well & steady)
    prox_increase = [[proximity_func(route[idx-1]) - proximity_func(node) for idx, node in enumerate(route[1:])] for route in sampled_routes]
    mean_prox_increase = [np.mean(prox) for prox in prox_increase]
    std_prox_increase = [np.std(prox) for prox in prox_increase]
    utility3 = [mean_prox_increase[i] / std_prox_increase[i] for i in range(len(sampled_routes))]
    utility3 = np.array(utility3) / np.sum(utility3)
    
    # Based on uniqueness of the route (how different from other routes)
    lcs_values = [[lcs_consecutive(route, route2) for route2 in sampled_routes if route2 != route] for route in sampled_routes]
    lcs_values = [np.mean(lcs) for lcs in lcs_values]
    utility4 = 1 / np.array(lcs_values)
    utility4 = utility4 / np.sum(utility4)

    # Merge all with some coefficients
    utilities = (coeffs[0] * utility1) + (coeffs[1] * utility2) + (coeffs[2] * utility3) + (coeffs[3] * utility4) #RK: Did you test if that actually works and allows to distinguish various kinds of paths? or is it augmented to one dimension and all properties are lost?
    return utilities


def _path_generator(network, origin, destination, proximity_func, beta, maxlen):
    path, current_node = list(), origin
    while True:
        path.append(current_node)
        options = [node for node in network.neighbors(current_node) if (node not in path)]
        if   (destination in options):                  return path + [destination]
        elif (not options) or (len(path) > maxlen):     return None
        else:       
            try:            
                current_node = _logit(options, proximity_func, beta)
            except:
                return None


def _logit(options, cost_function, beta):
    numerators = [np.exp(beta * cost_function(option)) for option in options]
    utilities = [numerator/sum(numerators) for numerator in numerators]
    return np.random.choice(options, p=utilities)


def lcs_non_consecutive(X, Y):
    """
    The LCS of two sequences is the longest subsequence that is present in both sequences in the same order, 
    but not necessarily consecutively.
    """
    m, n = len(X), len(Y)
    L = [[None]*(n+1) for i in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]


def lcs_consecutive(X, Y):
    """
    The LCS of two sequences is the longest subsequence that is present in both sequences in the same order, 
    consecutively.
    """
    m, n = len(X), len(Y)
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
    result = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i-1] == Y[j-1]:
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result

#################################################


################## FF Times #####################


def calculate_free_flow_time(route: list[str], network: nx.DiGraph) -> float:
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


################## To CSV #####################

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
            free_flow = calculate_free_flow_time(path, network)
            # Append the row to the DataFrame
            paths_df.loc[len(paths_df.index)] = [origin_name, dest_name, path_as_str, free_flow]
    return paths_df

#################################################


####################### Main #######################


def generate_paths(network: nx.DiGraph, origins: list[str], destinations: list[str], **kwargs) -> pd.DataFrame:
    params = get_params("params.json")
    
    for key, value in kwargs.items():
        params[key] = value

    number_of_paths = params[kc.NUMBER_OF_PATHS]
    beta = params[kc.BETA]
    weight = params[kc.WEIGHT]
    coeffs = params[kc.ROUTE_UTILITY_COEFFS]
    num_samples = params[kc.NUM_SAMPLES]
    max_path_length = params[kc.MAX_PATH_LENGTH]

    origins = {i : origin for i, origin in enumerate(origins)}
    destinations = {i : dest for i, dest in enumerate(destinations)}

    check_od_integrity(network, origins, destinations)
    routes = create_routes(network, number_of_paths, origins, destinations, beta, weight, coeffs, num_samples, max_path_length)
    paths_csv = paths_to_df(routes, network, origins, destinations)
    return paths_csv

#################################################