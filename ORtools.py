from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

class ORToolsSolver:
    def __init__(self, cvrp_instance):
        self.cvrp = cvrp_instance
        self.manager = pywrapcp.RoutingIndexManager(len(self.cvrp.cities), 
                                                  self.cvrp.num_vehicles,  # Using num_vehicles from CVRP
                                                  self.cvrp.depot_index)
        self.routing = pywrapcp.RoutingModel(self.manager)
        
    def distance_callback(self, from_index, to_index):
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return int(round(self.cvrp.distance_matrix[from_node][to_node]))
    
    def demand_callback(self, from_index):
        from_node = self.manager.IndexToNode(from_index)
        return self.cvrp.demands[from_node]
        
    def solve(self):
        transit_callback_index = self.routing.RegisterTransitCallback(self.distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(self.demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  
            [self.cvrp.capacity] * self.cvrp.num_vehicles,
            True,  
            'Capacity')
            
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.log_search = True
        search_parameters.solution_limit = 100000
        search_parameters.time_limit.seconds = 300  # 5 minutes
        search_parameters.guided_local_search_lambda_coefficient = 0.5
        solution = self.routing.SolveWithParameters(search_parameters)        
        if solution:
            total_cost = self.get_solution_cost(solution)
            return solution, total_cost
        return None, None   
    def get_solution_cost(self, solution):
        total_cost = 0
        for vehicle_id in range(self.cvrp.num_vehicles):
            route_cost = 0
            index = self.routing.Start(vehicle_id)
            while not self.routing.IsEnd(index):
                from_node = self.manager.IndexToNode(index)
                next_index = solution.Value(self.routing.NextVar(index))
                to_node = self.manager.IndexToNode(next_index)
                route_cost += self.cvrp.distance_matrix[from_node][to_node]
                index = next_index
            print(f"Vehicle {vehicle_id} route cost: {route_cost}")
            total_cost += route_cost
        return total_cost
    def get_solution_routes(self, solution):
        routes = []
        for vehicle_id in range(self.cvrp.num_vehicles):
            route = []
            index = self.routing.Start(vehicle_id)
            while not self.routing.IsEnd(index):
                route.append(self.manager.IndexToNode(index))
                index = solution.Value(self.routing.NextVar(index))
            route.append(self.manager.IndexToNode(index))
            routes.append(route)
        return routes