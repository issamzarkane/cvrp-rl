import numpy as np

class State:
    def __init__(self, cvrp):
        self.cvrp = cvrp
        self.visited = [1] * self.cvrp.num_cities
        self.visited[self.cvrp.depot_index] = 0
        self.current_vehicle = 0
        self.vehicle_loads = [0] * self.cvrp.num_vehicles
        
    def encode_state(self):
        encoded = self.visited.copy()
        encoded.extend([self.current_vehicle / self.cvrp.num_vehicles])
        encoded.extend([load / self.cvrp.capacity for load in self.vehicle_loads])
        return encoded
        
    def is_terminal(self):
        all_cities_visited = all(v == 0 for i, v in enumerate(self.visited) if i != self.cvrp.depot_index)
        all_vehicles_used = self.current_vehicle >= self.cvrp.num_vehicles
        return all_cities_visited or all_vehicles_used
        
    def transition(self, city):
        if self.visited[city] == 1:  # If city is unvisited
            self.visited[city] = 0
            # Update vehicle load when visiting a new city
            self.vehicle_loads[self.current_vehicle] += self.cvrp.demands[city]
            return True
        return False
    
    def is_action_valid(self, action):
        """Check if action is valid in current state"""
        # Check if action starts and ends at depot
        if action[0] != 0 or action[-1] != 0:
            return False
            
        # Check capacity constraint
        total_demand = sum(self.cvrp.demands[i] for i in action[1:-1])
        if total_demand > self.cvrp.capacity:
            return False
            
        # Check if action visits already visited cities
        for city in action[1:-1]:
            if self.visited[city] == 0:
                return False
                
        return True
    
    def get_unvisited_cities(self):
        """Return list of unvisited cities"""
        return [i for i, v in enumerate(self.visited) if v == 1]

    def copy(self):
        new_state = State(self.cvrp)
        new_state.visited = self.visited.copy()
        new_state.current_vehicle = self.current_vehicle
        new_state.vehicle_loads = self.vehicle_loads.copy()
        return new_state