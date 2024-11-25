import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), './')))

import inspect
import logging
import networkx as nx
import numpy as np
import pandas as pd

from typing import Callable

from path_generators import calculate_free_flow_time
from path_generators import paths_to_df
from path_generators.basic_generator import BasicPathGenerator
from utils import iterable_to_string

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

class HeuristicPathGenerator(BasicPathGenerator):
    """
    A path generator that incorporates heuristic-based scoring to select optimal paths
    between origin-destination pairs in a directed graph network.

    The HeuristicPathGenerator extends the functionality of the BasicPathGenerator
    by using heuristics and corresponding weights to evaluate and select paths from
    a set of sampled routes. It enables customizable scoring and deterministic selection
    of paths based on user-defined heuristics.
    Route generations can be made reproducible by setting a seed.

    Attributes:
        network (nx.DiGraph): The directed graph representing the network on which routes are generated.
        origins (list[str]): A list of origin node names in the network.
        destinations (list[str]): A list of destination node names in the network.
        heuristics (list[Callable]): A list of heuristic functions used to evaluate route groups.
            Each heuristic must be callable, accept `number_of_paths + 1` arguments (routes and the network),
            and return a numerical score deterministically.
        heur_weights (list[float]): A list of weights corresponding to each heuristic, determining
            the influence of each heuristic on the overall score.

    Methods (in addition to `BasicPathGenerator`):
        generate_routes() -> pd.DataFrame:
            Generates routes for all origin-destination pairs using heuristic-based selection
            and returns them in a structured DataFrame.
        _pick_routes_from_samples(sampled_routes: list[tuple]) -> list[tuple]:
            Selects a group of routes from a set of sampled routes based on heuristic scores.
        _validate_heuristics(sampled_routes: list[tuple]) -> None:
            Validates the heuristics to ensure they are callable, deterministic, and
            conform to the expected input and output structure.
    """
    def __init__(self, 
                 network: nx.DiGraph, 
                 origins: list[str], 
                 destinations: list[str], 
                 heuristics: list[Callable],
                 heur_weights: list[float],
                 **kwargs):
        
        super().__init__(network, origins, destinations, **kwargs)
        
        self.heuristics = heuristics
        self.heur_weights = heur_weights
        
        
    def generate_routes(self, as_df: bool = True) -> pd.DataFrame | dict:
        """
        Generates routes for all origin-destination pairs using heuristic-based selection.

        This method samples multiple paths between each origin-destination pair in the network.
        A set number of paths is sampled using a logit-based probabilistic approach, and then
        heuristics are applied to select the most optimal paths based on defined scoring criteria.

        Steps:
        1. For each destination node, calculate node potentials using shortest path lengths.
        2. For each origin node, sample a specified number of unique paths to the destination.
        3. Use heuristics and corresponding weights to evaluate and select the desired number of paths.
        4. Convert the selected routes into a structured DataFrame, if requested.
        
        Args:
            as_df (bool): A flag to determine whether to return the routes as a DataFrame or a dictionary.

        Returns:
            pd.DataFrame: A DataFrame containing the generated routes with the following columns:
                - `origins`: The origin node for each route.
                - `destinations`: The destination node for each route.
                - `path`: A string representation of the nodes in the route.
                - `free_flow_time`: The travel time for the route under free-flow conditions.

        Raises:
            AssertionError: 
                - If the number of samples is less than the number of desired routes to generate.
                - If the maximum path length is not greater than zero.
                - If the beta value is non-negative.
        """
        assert self.num_samples >= self.number_of_paths, f"Number of samples ({self.num_samples}) should be at least equal to the number of routes ({self.number_of_paths})"
        assert self.max_path_length > 0, f"Maximum path length should be greater than 0"
        assert self.beta < 0, f"Beta should be less than 0"
        
        routes = dict()   # Tuple<od_id, dest_id> : List<routes>
        for dest_idx, dest_name in self.destinations.items():
            node_potentials = dict(nx.shortest_path_length(self.network, target=dest_name, weight=self.weight))
            for origin_idx, origin_name in self.origins.items():
                sampled_routes = set()   # num_samples number of routes
                while (len(sampled_routes) < self.num_samples):
                    path = self._sample_single_route(origin_name, dest_name, node_potentials)
                    if not path is None:
                        sampled_routes.add(tuple(path))
                logging.info(f"Sampled {len(sampled_routes)} paths for {origin_idx} -> {dest_idx}")
                sampled_routes = sorted(list(sampled_routes), key=lambda x: iterable_to_string(x))
                routes[(origin_idx, dest_idx)] = self._pick_routes_from_samples(sampled_routes)
                logging.info(f"Selected {len(set(routes[(origin_idx, dest_idx)]))} paths for {origin_idx} -> {dest_idx}")
        if as_df:
            free_flows = {od: [calculate_free_flow_time(route, self.network) for route in routes[od]] for od in routes}
            routes_df = paths_to_df(routes, self.origins, self.destinations, free_flows)
            return routes_df
        else:
            return routes


    def _pick_routes_from_samples(self, sampled_routes: list[tuple]) -> list[tuple]:
        """
        Selects the most optimal set of routes from a list of sampled routes using heuristic scoring.

        This method evaluates groups of sampled routes based on a set of heuristics and their respective
        weights. It generates multiple groups of routes, calculates their scores using the heuristics,
        and selects the group with the highest overall score.

        Args:
            sampled_routes (list[tuple]): A list of sampled routes, where each route is represented 
                                        as a tuple of nodes.

        Returns:
            list[tuple]: A list of the selected routes, where each route is represented as a tuple of nodes.

        Raises:
            AssertionError:
                - If the desired number of paths to select exceeds the number of sampled routes.
                - If the number of paths to select (`number_of_paths`) is less than or equal to zero.
                - If any heuristic is not callable, does not accept the expected number of arguments,
                does not return a numerical value, or is non-deterministic.
                - If the number of heuristic weights does not match the number of heuristics.
        """
        assert self.number_of_paths <= len(sampled_routes), f"Number of paths ({self.number_of_paths}) should be less than or equal to the number of sampled routes ({len(sampled_routes)})"
        assert self.number_of_paths > 0, f"Number of paths should be greater than 0"
        self._validate_heuristics(sampled_routes)
        
        # Sample groups of routes (as their indices in the sampled_routes list)
        route_group_samples = set()
        while len(route_group_samples) < self.num_samples:
            sampled_group = self.rng.choice(len(sampled_routes), size=self.number_of_paths, replace=False)
            sampled_group = tuple(sorted(sampled_group))
            route_group_samples.add(sampled_group)
        route_group_samples = list(route_group_samples)
        
        # Calculate scores for each group of sampled paths
        scores = list()
        for sampled_group in route_group_samples:
            score = 0
            paths = [sampled_routes[i] for i in sampled_group]
            for heur_weight, heuristic in zip(self.heur_weights, self.heuristics):
                score += heur_weight * heuristic(*paths, self.network)
            scores.append(score)
        
        # Select the group of sampled paths with the highest score
        selected_paths = route_group_samples[np.argmax(scores)]
        selected_paths = [sampled_routes[i] for i in selected_paths]
        return selected_paths
    

    def _validate_heuristics(self, sampled_routes: list[tuple]) -> None:
        # Assert that each heuristic accepts as argument number_of_paths+1 items (paths and network)
        for heuristic in self.heuristics:
            # Check if the heuristic is callable
            assert callable(heuristic), f"Each heuristic must be callable, but found: {type(heuristic)}"
            # Check the number of arguments
            num_arguments = len(inspect.signature(heuristic).parameters)
            assert num_arguments == self.number_of_paths+1, f"Each heuristic must accept exactly {self.number_of_paths+1} arguments, but {heuristic.__name__} takes {num_arguments}."
            # Check if return is numerical
            example_return = heuristic(*sampled_routes[:self.number_of_paths], self.network)
            assert isinstance(example_return, (int, float)), f"Each heuristic must return a numerical value, but {heuristic.__name__} returns {type(example_return)}"
            # Check if the heuristic is deterministic
            assert heuristic(*sampled_routes[:self.number_of_paths], self.network) == heuristic(*sampled_routes[:self.number_of_paths], self.network), f"Each heuristic must be deterministic, but {heuristic.__name__} is not."
        assert len(self.heur_weights) == len(self.heuristics), f"Number of heuristic weights does not match with number of heuristics. ({len(self.heur_weights)} and {len(self.heuristics)})"
        
        
        
    def check_od_integrity(self):
        super().check_od_integrity()
        
    
    def _sample_single_route(self, origin: str, destination: str, node_potentials: dict) -> list[str] | None:
        return super()._sample_single_route(origin, destination, node_potentials)
    
    
    def _logit(self, options: list, node_potentials: dict) -> str:
        return super()._logit(options, node_potentials)