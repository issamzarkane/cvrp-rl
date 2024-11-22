# utils.py
from state import State
from cvrp import CVRP
from pyscipopt import Model
import torch
import matplotlib.pyplot as plt 

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




def visualize_route(cvrp_instance, route):
    """
    Visualizes the current policy's route on a 2D plane.

    Parameters:
        cvrp_instance (CVRP): The CVRP problem instance containing city coordinates.
        route (list): Sequential route as a list of city indices.
    """
    depot = cvrp_instance.cities[cvrp_instance.depot_index]
    cities = cvrp_instance.cities

    plt.figure(figsize=(10, 8))
    
    # Plot depot
    plt.scatter(depot[0], depot[1], c='red', label="Depot", s=100, zorder=5)
    plt.text(depot[0], depot[1], "Depot", fontsize=12, ha='right')
    
    # Plot cities
    for i, (x, y) in enumerate(cities):
        if i != cvrp_instance.depot_index:
            plt.scatter(x, y, c='blue', label="City" if i == 1 else "", s=50)
            plt.text(x, y, f"{i}", fontsize=10, ha='right')

    # Plot route
    for i in range(len(route) - 1):
        x1, y1 = cities[route[i]]
        x2, y2 = cities[route[i + 1]]
        plt.plot([x1, x2], [y1, y2], 'g-', linewidth=2)

    plt.title("Current Policy Route Visualization")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.grid()
    plt.show()

def solve_milp_with_value_function(cvrp_instance, state, value_net):
    """
    Solves the CVRP routing problem for the current state using SCIP,
    integrating immediate cost and cost-to-go (value function).

    Parameters:
        cvrp_instance (CVRP): The CVRP problem instance.
        state (State): Current state of the problem.
        value_net (ValueNetwork): Trained neural network for value approximation.

    Returns:
        list: Optimal route determined by SCIP.
    """
    n = cvrp_instance.num_cities
    distances = cvrp_instance.distance_matrix
    demands = cvrp_instance.demands
    Q = cvrp_instance.capacity

    # Initialize SCIP model
    model = Model("CVRP_with_Value_Function")

    # Decision variables
    x = {}  # Binary variables: x[i, j] = 1 if traveling from i to j
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

    u = {}  # Subtour elimination variables
    for i in range(1, n):
        u[i] = model.addVar(vtype="C", name=f"u_{i}")

    # Add capacity constraints
    load = {}
    for i in range(n):
        load[i] = model.addVar(vtype="C", name=f"load_{i}")

    # Objective: Minimize immediate cost + cost-to-go
    immediate_cost = sum(distances[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)
    state_tensor = torch.FloatTensor(state.encode_state())
    cost_to_go = value_net(state_tensor).item()
    model.setObjective(immediate_cost + cost_to_go, sense="minimize")

    # Constraints

    # Each city (except depot) must be visited exactly once
    for i in range(1, n):
        model.addCons(sum(x[i, j] for j in range(n) if i != j) == 1)
        model.addCons(sum(x[j, i] for j in range(n) if i != j) == 1)

    # Depot flow constraints
    model.addCons(sum(x[cvrp_instance.depot_index, j] for j in range(1, n)) == 1)
    model.addCons(sum(x[j, cvrp_instance.depot_index] for j in range(1, n)) == 1)

    # Subtour elimination constraints (MTZ formulation)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addCons(u[i] - u[j] + n * x[i, j] <= n - 1)

    # Capacity constraints
    for i in range(1, n):
        model.addCons(u[i] <= Q)
        model.addCons(u[i] >= demands[i])

    # Vehicle capacity constraints
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addCons(load[j] >= load[i] + demands[j] * x[i,j] 
                            - Q * (1 - x[i,j]))
    
    # Load bounds
    for i in range(1, n):
        model.addCons(demands[i] <= load[i] <= Q)

    # Optimize
    model.optimize()

    # Extract the optimal route
    route = []
    if model.getStatus() == "optimal":
        for i in range(n):
            for j in range(n):
                if i != j and model.getVal(x[i, j]) > 0.5:
                    route.append((i, j))
    return route

def extract_sequential_route(edges, start_node=0):
    """
    Converts a list of (i, j) edges into a sequential route.

    Parameters:
        edges (list of tuples): List of edges (i, j) representing the route.
        start_node (int): Starting node, typically the depot (0).

    Returns:
        list: Sequential route as a list of nodes.
    """
    route = [start_node]
    current_node = start_node

    while len(route) <= len(edges):  # Ensure all edges are included
        for i, j in edges:
            if i == current_node:  # Find the next edge starting from the current node
                route.append(j)
                current_node = j
                break
    return route