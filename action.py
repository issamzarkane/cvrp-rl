import itertools

class Action:
    def __init__(self, cvrp, state):
        self.cvrp = cvrp
        self.state = state
        
    def get_all_actions(self):
        """Generate all feasible actions (routes) from current state"""
        feasible_actions = []
        unvisited = [i for i, v in enumerate(self.state.visited) if v == 1]
        
        # Get feasible routes for current vehicle
        current_load = self.state.vehicle_loads[self.state.current_vehicle]
        
        for length in range(1, len(unvisited) + 1):
            for subset in itertools.combinations(unvisited, length):
                # Try all permutations of the subset
                for perm in itertools.permutations(subset):
                    route = [self.cvrp.depot_index] + list(perm) + [self.cvrp.depot_index]
                    route_demand = sum(self.cvrp.demands[i] for i in perm)
                    
                    # Check if route is feasible for current vehicle
                    if current_load + route_demand <= self.cvrp.capacity:
                        feasible_actions.append(route)
                        
        # If no feasible actions for current vehicle, try next vehicle
        if not feasible_actions and self.state.current_vehicle < self.cvrp.num_vehicles - 1:
            self.state.current_vehicle += 1
            return self.get_all_actions()
            
        return feasible_actions        