import networkx as nx
import pandas as pd



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