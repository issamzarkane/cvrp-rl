from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

class ORToolsSolver:
    def __init__(self, cvrp_instance):
        self.cvrp = cvrp_instance
        self.manager = pywrapcp.RoutingIndexManager(len(self.cvrp.cities), 
                                                  1,  # Number of vehicles
                                                  0)  # Depot
        self.routing = pywrapcp.RoutingModel(self.manager)
        
    def distance_callback(self, from_index, to_index):
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.cvrp.distance_matrix[from_node][to_node]
    
    def demand_callback(self, from_index):
        from_node = self.manager.IndexToNode(from_index)
        return self.cvrp.demands[from_node]
        
    def solve(self):
        # Register callbacks
        transit_callback_index = self.routing.RegisterTransitCallback(self.distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity constraint
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(self.demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [self.cvrp.capacity],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')
            
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            
        # Solve
        solution = self.routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self.get_solution_cost(solution)
        return None
        
    def get_solution_cost(self, solution):
        total_cost = 0
        index = self.routing.Start(0)
        while not self.routing.IsEnd(index):
            previous_index = index
            index = solution.Value(self.routing.NextVar(index))
            total_cost += self.routing.GetArcCostForVehicle(previous_index, index, 0)
        return total_cost
