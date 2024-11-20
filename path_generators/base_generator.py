import networkx as nx

class PathGenerator:
    """
    A base class for generating routes in a directed graph network.

    Attributes:
        network (nx.DiGraph): The directed graph representing the network on which routes are generated.

    Methods:
        generate_routes():
            Abstract method to be implemented by subclasses for generating routes.
            Raises NotImplementedError if not overridden.
    """
    
    def __init__(self, 
                 network: nx.DiGraph):
        """
        Initializes the PathGenerator with a directed graph network.

        Args:
            network (nx.DiGraph): A directed graph representing the network.
        """
        self.network = network


    def generate_routes(self):
        """
        Abstract method for generating routes.

        This method should be implemented by subclasses to define specific 
        route generation logic.

        Raises:
            NotImplementedError: If called on the base class without being overridden.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")