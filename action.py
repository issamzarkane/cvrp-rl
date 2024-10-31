import itertools

class Action :
    def __init__(self, cvrp_instance, state) : 
        self.cvrp = cvrp_instance
        self.num_cities_no_depot = cvrp_instance.num_cities - 1
        self.state = state

    def generate_partial_permutations(self,k) :
        # generate possible permutations of size k of only unvisited cities
        unvisited_cities = self.state.get_unvisited_cities() 
        all_permutations = itertools.permutations(unvisited_cities, k)

        feasible_routes = []
        for route in all_permutations :
            if self.is_feasible_route(route) :
                full_route = (0,) + route + (0,) #Add depot in start and finish
                feasible_routes.append(full_route)
        return feasible_routes
    
    def get_all_actions(self) :
        # generate all actions (permutations) of sizes 1 to num_cities
        all_actions = []
        for k in range(1,self.num_cities_no_depot + 1) : # 1 to n-1
            all_actions.extend(self.generate_partial_permutations(k))
        return all_actions 
    
    def is_feasible_route(self,route) :
        route_demand = sum(self.cvrp.demands[city] for city in route)
        if route_demand > self.cvrp.capacity :
            return False
        return all(self.state.visited[city] == 1 for city in route)