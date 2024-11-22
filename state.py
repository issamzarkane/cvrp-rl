import numpy as np

class State:
    def __init__(self, cvrp_instance, visited=None):
        self.cvrp = cvrp_instance
        # Initialize visited cities array (0 for unvisited, 1 for visited)
        if visited is None:
            self.visited = [0] * len(self.cvrp.cities)
            self.visited[0] = 1  # Mark depot as visited
        else:
            self.visited = visited
            
    def encode_state(self):
        """Binary encoding of state as described in the paper"""
        return np.array(self.visited, dtype=np.float32)
    
    def transition(self, action):
        """Execute action and return new state"""
        if not self.is_action_valid(action):
            return None
            
        new_visited = self.visited.copy()
        for city in action:
            if city != 0:  # Skip depot
                new_visited[city] = 1
                
        return State(self.cvrp, new_visited)
    
    def is_terminal(self):
        """Check if all cities have been visited"""
        return all(self.visited[1:])  # Exclude depot
    
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
            if self.visited[city] == 1:
                return False
                
        return True
    
    def get_unvisited_cities(self):
        """Return list of unvisited cities"""
        return [i for i, v in enumerate(self.visited) if v == 0]
