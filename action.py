import itertools

class Action :
    def __init__(self, cvrp_instance) : 
        self.cvrp = cvrp_instance
        self.num_cities_no_depot = cvrp_instance.num_cities - 1

    def generate_partial_permutations(self,k) :
        # generate possible permutations of size k 
        cities = list(range(1, self.num_cities_no_depot + 1))
        all_permutations = itertools.permutations(cities, k)

        feasible_routes = []
        for route in all_permutations :
            if self.is_feasible_route(route) :
                feasible_routes.append(route)
        return feasible_routes
    
    def get_all_actions(self) :
        # generate all actions (permutations) of size 1 to num_cities
        all_actions = []
        for k in range(1,self.num_cities_no_depot + 1) : # 1 to n-1
            all_actions.extend(self.generate_partial_permutations(k))
        return all_actions 
    
    def is_feasible_route(self,route) :
        route_demand = sum(self.cvrp.demands[city] for city in route)
        return route_demand <= self.cvrp.capacity