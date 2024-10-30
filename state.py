class State:
    def __init__(self, cvrp_instance) :
        self.cvrp = cvrp_instance
        self.num_cities = cvrp_instance.num_cities 
        # Initial state : depot + cities are all marked 1 (not visited)
        self.visited = [1] * cvrp_instance.num_cities  #  = [1,1,1,....,1] 

    def transition(self, city_index) : 
        #transition(i) means we go to city i e.i. setting its index in visited to 0
        if city_index < self.num_cities and city_index != self.cvrp.depot_index and self.visited[city_index] == 1 : 
            self.visited[city_index] = 0 

    def is_terminal(self) : 
        #Checking if we visited all the cities, True is state is terminal, False if not
        return all(v==0 for v in self.visited[1:]) 
    
    def get_unvisited_cities(self) :
        # returns the indexes where the value is 1 besides the depot
        return [i for i, visited in enumerate(self.visited) if visited == 1 and i!=self.cvrp.depot_index]
    
    def encode_state(self) :
        return tuple(self.visited)