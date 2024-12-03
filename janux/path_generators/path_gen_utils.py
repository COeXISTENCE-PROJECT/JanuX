import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './'))

import networkx as nx
import pandas as pd

from typing import Union

from janux.utils import iterable_to_string



def calculate_free_flow_time(route: list[str], network: nx.DiGraph) -> float:
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



def check_od_integrity(network: nx.DiGraph, origins: list[str], destinations: list[str]):
    """
    Validates the integrity of origin-destination pairs in the network.

    This method ensures that:
    1. All origin and destination nodes are present in the network.
    2. Each origin node can reach all specified destination nodes.

    Raises:
        AssertionError: If an origin or destination node is missing from the network or if an origin cannot reach a destination.
    """
    for origin in origins:
        assert origin in network.nodes, f"Origin {origin} is not in the network"
    for destination in destinations:
        assert destination in network.nodes, f"Destination {destination} is not in the network."
        
    for origin in origins:
        paths_from_origin = nx.multi_source_dijkstra_path(network, sources=[origin])
        for destination in destinations:
            assert destination in paths_from_origin, f"Origin {origin} cannot reach destination {destination}."



def paths_to_df(routes: dict, origins: dict, destinations: dict, free_flows: Union[dict, None]=None) -> pd.DataFrame:
    # Assert that `routes` and `free_flows` has the same structure
    if free_flows is not None:
        assert routes.keys() == free_flows.keys()
        for key in routes.keys():
            assert len(routes[key]) == len(free_flows[key])
    # Initialize an empty DataFrame with the required columns
    columns = ["origins", "destinations", "path"]
    if free_flows is not None:
        columns.append('free_flow_time')
    paths_df = pd.DataFrame(columns=columns)
    
    # Iterate through the routes dictionary
    for (origin_idx, dest_idx), paths in routes.items():
        # Retrieve node names of the OD
        origin_name = origins[origin_idx]
        dest_name = destinations[dest_idx]
        for path_idx, path in enumerate(paths):
            # Convert the path to a string format
            path_as_str = iterable_to_string(path, ",")
            if free_flows is not None:
                # Retrieve the free-flow travel time for the path
                free_flow = free_flows[(origin_idx, dest_idx)][path_idx]
                # Append the row to the DataFrame
                paths_df.loc[len(paths_df.index)] = [origin_name, dest_name, path_as_str, free_flow]
            else:
                paths_df.loc[len(paths_df.index)] = [origin_name, dest_name, path_as_str]
    return paths_df