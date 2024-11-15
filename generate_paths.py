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
    free_flows = [_get_ff(route, network) for route in sampled_routes]
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

def _get_ff(path, network):
    """ Get ff time for a given route """
    length = pd.DataFrame(network.edges(data = True))
    time = length[2].astype('str').str.split(':',expand=True)[1]
    length[2] = time.str.replace('}','',regex=True).astype('float')
    rou=[]
    for i in range(len(path)):
        if i < len(path) - 1:
            for k in range(len(length[0])):
                if (path[i] == length[0][k]) and (path[i + 1] == length[1][k]):
                    rou.append(length[2][k])
    return sum(rou)

#################################################


################## Disk Ops #####################

def paths_to_df(routes, network, origins, destinations):
    """ Make a dataframe from the routes """
    paths_df = pd.DataFrame(columns = [kc.ORIGINS, kc.DESTINATIONS, kc.PATH, kc.FREE_FLOW_TIME])
    for od, paths in routes.items():
        for path in paths:
            paths_df.loc[len(paths_df.index)] = [origins[od[0]], destinations[od[1]], list_to_string(path, ","), _get_ff(path, network)]
    return paths_df

#################################################


####################### Main #######################


def generate_paths(network: nx.classes.digraph.DiGraph, origins: list[str], destinations: list[str], **kwargs) -> pd.DataFrame:
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
    df_to_prettytable(paths_csv)
    return paths_csv

#################################################