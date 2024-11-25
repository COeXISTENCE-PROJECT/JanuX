import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import logging
import networkx as nx
import numpy as np
import pandas as pd

from keychain import Keychain as kc
from path_generators import calculate_free_flow_time
from path_generators import check_od_integrity
from path_generators import paths_to_df
from path_generators.base_generator import PathGenerator
from utils import get_params
from utils import iterable_to_string

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

class BasicPathGenerator(PathGenerator):
    """
    A path generation class for creating and sampling routes in a directed graph network.

    The BasicPathGenerator extends the PathGenerator base class to include functionality
    for generating paths between origin-destination pairs in a network. It uses shortest paths,
    sampling methods, and logit-based probabilistic models to select and generate routes.
    Route generations can be made reproducible by setting a seed.

    Attributes:
        origins (dict): A dictionary mapping origin indices to their node names in the network.
        destinations (dict): A dictionary mapping destination indices to their node names in the network.
        number_of_paths (int): Number of distinct paths to generate for each origin-destination pair.
        beta (float): Parameter controlling the sensitivity to node potentials in the logit model.
        weight (str): Edge attribute used as the weight for calculating shortest paths.
        num_samples (int): Number of paths to sample before selecting final paths.
        max_path_length (int): Maximum allowed length for a generated path.
        random_seed (int or None): Random seed for reproducibility in sampling.
        rng (np.random.Generator): Random number generator for sampling.

    Methods:
        check_od_integrity():
            Ensures that all origins and destinations are present in the network and are reachable.
        generate_routes() -> pd.DataFrame:
            Generates paths for all origin-destination pairs and returns them as a DataFrame.
        _sample_single_route(origin: str, destination: str, node_potentials: dict) -> list[str] | None:
            Samples a single path probabilistically from an origin to a destination.
        _pick_routes_from_samples(sampled_routes: list[tuple]) -> list[tuple]:
            Selects the desired number of paths from a set of sampled routes based on their frequencies.
        _logit(options: list, node_potentials: dict) -> str:
            Selects a node probabilistically based on logit probabilities using node potentials.
    """
    def __init__(self, 
                 network: nx.DiGraph, 
                 origins: list[str], 
                 destinations: list[str], 
                 **kwargs):
        
        super().__init__(network)
        check_od_integrity(self.network, origins, destinations)

        # Convert origin and destination names to indices
        self.origins = dict(enumerate(origins))
        self.destinations = dict(enumerate(destinations))
        
        # Get parameters from the params.json file and update them with the provided kwargs
        params = get_params("path_generators/path_gen_params.json")
        params.update(kwargs)
        
         # Get parameters
        self.number_of_paths = params[kc.NUMBER_OF_PATHS]
        self.beta = params[kc.BETA]
        self.weight = params[kc.WEIGHT]
        self.num_samples = params[kc.NUM_SAMPLES]
        self.max_path_length = params[kc.MAX_PATH_LENGTH]
        if self.max_path_length is None:
            self.max_path_length = float("inf")
        
        # Set random seed if provided
        self.random_seed = params.get(kc.RANDOM_SEED, None)
        np.random.seed(self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        
        
    def generate_routes(self, as_df: bool = True) -> pd.DataFrame | dict:
        """
        Generates paths for all origin-destination (OD) pairs in the network.

        This method samples multiple paths between each origin and destination pair
        using a probabilistic approach. It selects the desired number of unique routes
        based on sampling probabilities, converts them into a structured format, 
        and returns them as a DataFrame or a dictionary.
        
        Args:
            as_df (bool): A flag to determine whether to return the routes as a DataFrame or a dictionary. 

        Returns:
            pd.DataFrame: A DataFrame containing the generated routes with the following columns:
                - `origins`: The origin node for each route.
                - `destinations`: The destination node for each route.
                - `path`: A string representation of the nodes in the route.
                - `free_flow_time`: The travel time for the route under free-flow conditions.

        Raises:
            AssertionError: If the number of samples is less than the number of paths 
                            to be generated, or if the maximum path length is not positive,
                            or beta is non-negative.
        """
        assert self.num_samples >= self.number_of_paths, f"Number of samples ({self.num_samples}) should be \
            at least equal to the number of routes ({self.number_of_paths})"
        assert self.max_path_length > 0, f"Maximum path length should be greater than 0"
        assert self.beta < 0, f"Beta should be less than 0"
        
        routes = dict()   # Tuple<od_id, dest_id> : List<routes>
        for dest_idx, dest_name in self.destinations.items():
            node_potentials = dict(nx.shortest_path_length(self.network, target=dest_name, weight=self.weight))
            for origin_idx, origin_name in self.origins.items():
                sampled_routes = list()   # num_samples number of routes
                while (len(sampled_routes) < self.num_samples) or (len(set(sampled_routes)) < self.number_of_paths):
                    path = self._sample_single_route(origin_name, dest_name, node_potentials)
                    if not path is None:
                        sampled_routes.append(tuple(path))
                logging.info(f"Sampled {len(sampled_routes)} paths for {origin_idx} -> {dest_idx}")
                routes[(origin_idx, dest_idx)] = self._pick_routes_from_samples(sampled_routes)
                logging.info(f"Selected {len(set(routes[(origin_idx, dest_idx)]))} paths for {origin_idx} -> {dest_idx}")
        if as_df:
            free_flows = {od: [calculate_free_flow_time(route, self.network) for route in routes[od]] for od in routes}
            routes_df = paths_to_df(routes, self.origins, self.destinations, free_flows)
            return routes_df
        else:
            return routes


    def _sample_single_route(self, origin: str, destination: str, node_potentials: dict) -> list[str] | None:
        """
        Samples a single path probabilistically from the origin to the destination.

        This method iteratively builds a path by selecting the next node from the neighbors
        of the current node using a logit-based probabilistic model. The sampling process
        ends if:
        1. The destination is reached.
        2. There are no valid next nodes to continue the path.
        3. The maximum allowable path length is exceeded.

        Args:
            origin (str): The starting node of the path.
            destination (str): The target node of the path.
            node_potentials (dict): A dictionary mapping node names to their potentials,
                                    used to calculate the selection probabilities.

        Returns:
            list[str] | None: A list representing the nodes in the sampled path if successful,
                            or None if a valid path could not be found.
        """
        path, current_node = list(), origin
        while True:
            path.append(current_node)
            options = [node for node in sorted(self.network.neighbors(current_node)) if node not in path]
            if   (destination in options):                  return path + [destination]
            elif (not options) or (len(path) > self.max_path_length):     return None
            else:       
                try:            
                    current_node = self._logit(options, node_potentials)
                except:
                    return None
    
    
    def _pick_routes_from_samples(self, sampled_routes: list[tuple]) -> list[tuple]:
        """
        Selects the desired number of unique routes from a set of sampled routes.

        This method filters through a list of sampled routes to pick a specified number
        of unique paths. The selection is based on the frequency of occurrence of each
        unique route, using a probability distribution derived from their counts.

        Args:
            sampled_routes (list[tuple]): A list of sampled routes, where each route is 
                                        represented as a tuple of nodes.

        Returns:
            list[tuple]: A list of selected unique routes, with the number of routes
                        equal to the specified `number_of_paths`.

        Raises:
            AssertionError: If the number of paths to select exceeds the total number
                            of sampled routes or the number of unique routes.
        """
        assert self.number_of_paths <= len(sampled_routes), f"Number of paths ({self.number_of_paths}) should be less than or equal to the number of sampled routes ({len(sampled_routes)})"
        assert self.number_of_paths > 0, f"Number of paths should be greater than 0"
        
        sampled_routes_by_str = np.array([iterable_to_string(route, ",") for route in sampled_routes])
        # Get each unique route and their counts
        unique_routes, route_counts = np.unique(sampled_routes_by_str, return_counts=True)
        # Calculate sampling probabilities (according to their counts)
        sampling_probabilities = route_counts / route_counts.sum()
        # Sample from the unique items according to the probabilities
        assert self.number_of_paths <= len(unique_routes), f"Cannot sample {self.number_of_paths} distinct items from {len(unique_routes)} unique items."
        picked_routes = self.rng.choice(unique_routes, size=self.number_of_paths, p=sampling_probabilities, replace=False)
        picked_routes = [tuple(route.split(",")) for route in picked_routes]
        return picked_routes


    def _logit(self, options: list, node_potentials: dict) -> str:
        # If a node does not have a potential, it is a dead end, so we assign an infinite potential
        numerators = [np.exp(self.beta * node_potentials.get(option, float("inf"))) for option in options]
        utilities = [numerator/sum(numerators) for numerator in numerators]
        choice = str(self.rng.choice(options, p=utilities))
        return choice