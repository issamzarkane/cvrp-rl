import numpy as np 
from scipy.spatial import distance_matrix
from numpy import random
from state import State

class CVRP:
    def __init__(self, cities, demands, capacity, num_vehicles, depot_index=0):
        # Added num_vehicles parameter
        self.cities = cities
        self.demands = demands
        self.capacity = capacity
        self.num_vehicles = num_vehicles  # New attribute
        self.depot_index = depot_index
        self.num_cities = len(cities)

        self.distance_matrix = distance_matrix(self.cities, self.cities)

    def get_demand(self, city_index):
        return self.demands[city_index]
    
    def is_feasible_route(self, route):
        # Check if route is feasible for any vehicle
        route_demand = sum(self.demands[city] for city in route if city != self.depot_index)
        return route_demand <= self.capacity
    
    def sample_random_start_state(self):
        random_state = State(self)
        num_cities = len(self.cities) - 1
        # Randomly assign some cities as visited
        num_visited = random.randint(1, num_cities)
        visited_cities = random.sample(range(1, num_cities + 1), num_visited)
        for city in visited_cities:
            random_state.transition(city)
        return random_state

    def get_total_demand(self):
        # New helper method to get total demand of all cities
        return sum(self.demands[i] for i in range(1, self.num_cities))

    def verify_instance(self):
        # New method to verify instance feasibility
        total_demand = self.get_total_demand()
        total_capacity = self.capacity * self.num_vehicles
        return total_capacity >= total_demand
        return random_state