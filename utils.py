# utils.py
from state import State
from cvrp import CVRP

def transition(state, action):
    """
    Deterministic transition function T(s, a) that returns a new state 
    after taking action a, marking cities in the route as visited.
    
    Parameters:
        state (State): The current state before taking the action.
        action (tuple): The route (action) to take, starting and ending with the depot.
        
    Returns:
        State: A new State instance with updated visited status.
        None: If action is invalid (i.e., revisits any city already visited in state).
    """
    # Check if the action includes any city that has already been visited
    if any(state.visited[city] == 0 for city in action if city != state.cvrp.depot_index):
        return None  # Invalid action; revisits a city already visited

    # Create a new state instance as a copy of the current state
    new_state = State(state.cvrp)
    new_state.visited = state.visited[:]  # Copy the visited list to avoid modifying the original

    # Mark cities in the action as visited
    for city in action:
        if city != state.cvrp.depot_index:  # Exclude the depot from marking
            new_state.visited[city] = 0

    return new_state


def cost(cvrp, action):
    """
    Cost function C(a) that calculates the total distance of a given action route.

    Parameters:
        cvrp (CVRP): The CVRP instance containing the distance matrix.
        action (tuple): The route (action), represented as a sequence of city indices,
                        starting and ending at the depot.

    Returns:
        float: The total travel distance of the route.
    """
    total_cost = 0
    for i in range(len(action) - 1):
        total_cost += cvrp.distance_matrix[action[i]][action[i + 1]]
    return total_cost
