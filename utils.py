# utils.py
from state import State
from cvrp import CVRP
import torch
import matplotlib.pyplot as plt
from pyscipopt import Model

def transition(state, action):
    """
    Deterministic transition function T(s, a) that returns a new state 
    after taking action a, marking cities in the route as visited.
    """
    # Check if action is valid for current vehicle
    route_demand = sum(state.cvrp.demands[city] for city in action if city != state.cvrp.depot_index)
    current_load = state.vehicle_loads[state.current_vehicle]
    
    if current_load + route_demand > state.cvrp.capacity:
        return None  # Invalid action; exceeds vehicle capacity
        
    if any(state.visited[city] == 0 for city in action if city != state.cvrp.depot_index):
        return None  # Invalid action; revisits a city
        
    # Create new state
    new_state = State(state.cvrp)
    new_state.visited = state.visited[:]
    new_state.current_vehicle = state.current_vehicle
    new_state.vehicle_loads = state.vehicle_loads[:]
    
    # Update state
    for city in action:
        if city != state.cvrp.depot_index:
            new_state.visited[city] = 0
    
    # Update vehicle load
    new_state.vehicle_loads[state.current_vehicle] += route_demand
    
    # Move to next vehicle if current route completed
    if action[-1] == state.cvrp.depot_index:
        new_state.current_vehicle = min(state.current_vehicle + 1, state.cvrp.num_vehicles - 1)
        
    return new_state

def solve_milp_with_value_function(cvrp_instance, state, value_net):
    """
    Solves CVRP using MILP with value function approximation.
    """
    model = Model("CVRP_with_Value_Function")
    n = cvrp_instance.num_cities
    
    # Decision variables for each vehicle
    x = {}
    for v in range(cvrp_instance.num_vehicles):
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[v,i,j] = model.addVar(vtype="B", name=f"x_{v}_{i}_{j}")
    
    # Add vehicle-specific constraints
    for v in range(cvrp_instance.num_vehicles):
        # Flow conservation
        for i in range(1, n):
            model.addCons(
                sum(x[v,i,j] for j in range(n) if j != i) == 
                sum(x[v,j,i] for j in range(n) if j != i)
            )
        
        # Capacity constraints
        model.addCons(
            sum(cvrp_instance.demands[i] * x[v,i,j] 
                for i in range(1, n) 
                for j in range(n) if i != j) <= cvrp_instance.capacity
        )
    
    # Each city must be visited exactly once by any vehicle
    for i in range(1, n):
        model.addCons(
            sum(x[v,i,j] 
                for v in range(cvrp_instance.num_vehicles)
                for j in range(n) if j != i) == 1
        )
    
    # Solve and return solution
    return model.optimize()

def visualize_route(cvrp_instance, route):
    """
    Visualizes the current policy's route on a 2D plane.
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