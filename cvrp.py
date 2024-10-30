import numpy as np 
from scipy.spatial import distance_matrix

class CVRP :
    def __init__(self, cities, demands, capacity, depot_index=0) :
        #Example : 
        # CVRP(
        # [[1,2],[6,5],[5,7]] :  depot in (1,2) and 2 cities that we need to visit (6,5) & (5,7)
        # [2,3] : 2 is the demand for the first city and 3 is the demand of the third city (maybe add 0 for depot demand for simplicity)
        # 6 : here 6 is the capacity (make sure Q > sum(d_i) in this example 6 > 2+3)
        # depot_index = 0 for clarity and maybe can be used later in code (may be deleted if no necessary) 
        #)
        self.cities = cities
        self.demands = demands
        self.capacity = capacity 
        self.depot_index = depot_index
        self.num_cities = len(cities) #including the depot

        self.distance_matrix = distance_matrix(self.cities,self.cities)
        #in our example our distance matrix would be : ([[0.        , 5.83095189, 6.40312424],
        #                                                [5.83095189, 0.        , 2.23606798],
        #                                                [6.40312424, 2.23606798, 0.        ]])

    def get_demand(self, city_index) :
        return self.demands[city_index]
    
    def is_feasible_route(self, route):
        route_demand = sum(self.demands[city] for city in route if city != self.depot_index)
        return route_demand <= self.capacity