import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import logging
import networkx as nx
import numpy as np
import pandas as pd

from typing import List, Union

from janux.path_generators import calculate_free_flow_time
from janux.path_generators import check_od_integrity
from janux.path_generators import paths_to_df
from janux.path_generators.base_generator import PathGenerator
from janux.utils import get_params
from janux.utils import iterable_to_string

class BasicPathGenerator(PathGenerator):

    """
    Generates paths in a transportation network based on given origins and destinations.

    This class uses a probabilistic approach to sample routes between specified origins 
    and destinations in a directed graph (network). The sampling is based on a logit model.

    Args:
        network (nx.DiGraph): The transportation network represented as a directed graph.
        origins (list[str]): A list of origin nodes in the network.
        destinations (list[str]): A list of destination nodes in the network.
        **kwargs: Additional parameters to customize path generation. Overrides defaults 
                  in `path_gen_params.json`.

    Attributes:
        origins (dict): A mapping of origin indices to their names.
        destinations (dict): A mapping of destination indices to their names.
        number_of_paths (int): Number of unique paths to generate per origin-destination pair.
        beta (float): Logit model parameter; controls sensitivity to potential differences.
        weight (str): Edge attribute used as the cost or weight for shortest-path calculations.
        num_samples (int): Number of routes to sample before selecting unique paths.
        random_seed (int | None): Seed for reproducible random number generation.
        rng (np.random.Generator): Random number generator instance.
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
        
        # Determine the absolute path to the `path_gen_params.json` file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        params_file_path = os.path.join(current_dir, "path_gen_params.json")
        
        # Get parameters from the params.json file and update them with the provided kwargs
        params = get_params(params_file_path)
        params.update(kwargs)
        
        # Get parameters
        self.number_of_paths = params["number_of_paths"]
        self.beta = params["beta"]
        self.weight = params["weight"]
        self.num_samples = params["num_samples"]
        
        # Set random seed if provided
        self.random_seed = params.get("random_seed", None)
        np.random.seed(self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        
        self.verbose = params["verbose"]
        self.logger = logging.getLogger(__name__)
        if self.logger.hasHandlers():   self.logger.handlers.clear()
        if self.verbose:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.CRITICAL + 1)
        
        
    def generate_routes(self, as_df: bool = True, calc_free_flow: bool = False) -> Union[pd.DataFrame, dict]:
        
        """
        Generates routes between origin-destination pairs in the network.

        This method samples a specified number of routes for each origin-destination pair 
        using a probabilistic logit model and selects unique paths from the sampled routes. 
        The results can be returned either as a DataFrame or a dictionary.

        Args:
            as_df (bool): If True, the routes are returned as a pandas DataFrame. 
                        If False, the routes are returned as a dictionary. 
                        Defaults to True.
            calc_free_flow (bool): If True, the free-flow travel time for each route is calculated.

        Returns:
            pd.DataFrame | dict: 
                - A DataFrame containing the routes and associated metadata if `as_df` is True.
                - A dictionary mapping (origin_id, destination_id) tuples to lists of selected 
                routes if `as_df` is False.

        Raises:
            AssertionError: If `num_samples` is less than `number_of_paths`.
            AssertionError: If `beta` is not less than 0.

        Notes:
            - Each route is represented as a list of node names.
            - The sampling process ensures that the specified number of unique paths 
            (`number_of_paths`) is selected for each origin-destination pair.
        """
        
        assert self.num_samples >= self.number_of_paths, f"Number of samples ({self.num_samples}) should be \
            at least equal to the number of routes ({self.number_of_paths})"
        assert self.beta < 0, f"Beta should be less than 0"
        
        routes = dict()   # Tuple<od_id, dest_id> : List<routes>
        for dest_idx, dest_name in self.destinations.items():
            node_potentials = dict(nx.shortest_path_length(self.network, target=dest_name, weight=self.weight))
            for origin_idx, origin_name in self.origins.items():
                sampled_routes = list()   # num_samples number of routes
                while (len(sampled_routes) < self.num_samples) or (len(set(sampled_routes)) < self.number_of_paths):
                    path = self._sample_single_route(origin_name, dest_name, node_potentials)
                    sampled_routes.append(tuple(path))
                self.logger.info(f"Sampled {len(sampled_routes)} paths for {origin_idx} -> {dest_idx}")
                routes[(origin_idx, dest_idx)] = self._pick_routes_from_samples(sampled_routes)
                self.logger.info(f"Selected {len(set(routes[(origin_idx, dest_idx)]))} paths for {origin_idx} -> {dest_idx}")
                
        if as_df:
            free_flows = None
            if calc_free_flow:
                free_flows = {od: [calculate_free_flow_time(route, self.network) for route in routes[od]] for od in routes}
            routes_df = paths_to_df(routes, self.origins, self.destinations, free_flows)
            return routes_df
        else:
            return routes


    def _sample_single_route(self, origin: str, destination: str, node_potentials: dict) -> Union[List[str], None]:
        
        """
        Samples a single route between an origin and destination using a logit-based probabilistic model.

        This method constructs a path iteratively by selecting the next node based on 
        a logit model that uses node potentials as weights. The process stops when the 
        destination is reached or no valid route exists.

        Args:
            origin (str): The starting node of the route.
            destination (str): The target node of the route.
            node_potentials (dict): A dictionary mapping nodes to their potentials, 
                                    where potentials represent the "attractiveness" of a node 
                                    as a step toward the destination.

        Returns:
            list[str] | None: 
                - A list of nodes representing the path from the origin to the destination 
                if a route is successfully sampled.
                - None if no valid route exists.

        Notes:
            - The method selects nodes probabilistically using a logit model, where the likelihood 
            of choosing a node is proportional to the exponential of its potential scaled by beta.
            - If the destination is a direct neighbor of the current node, the route is completed.
            - If no valid options are available at any step, the route construction fails.
        """
        
        path, current_node = list(), origin
        while True:
            path.append(current_node)
            options = sorted(self.network.neighbors(current_node))
            if destination in options:
                return path + [destination]
            else:
                current_node = self._logit(options, node_potentials)
    
    
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