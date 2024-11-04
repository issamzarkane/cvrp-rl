import numpy as np
import torch
import matplotlib.pyplot as plt
from cvrp import CVRP
from state import State
from action import Action
from policy import Policy, ValueNetwork, policy_iteration
from utils import transition, cost

def visualize_route(cvrp_instance, route):
    """
    Visualizes the CVRP route on a 2D plane with depot and cities.
    """
    depot = cvrp_instance.cities[cvrp_instance.depot_index]
    cities = cvrp_instance.cities[1:]  # Exclude depot for plotting

    plt.figure(figsize=(8, 8))
    
    # Plot depot
    plt.plot(depot[0], depot[1], 'rs', markersize=10, label="Depot")
    
    # Plot cities
    for i, (x, y) in enumerate(cities, start=1):
        plt.plot(x, y, 'bo')
        plt.text(x + 0.2, y + 0.2, f'{i}', fontsize=12)
    
    # Plot route
    for i in range(len(route) - 1):
        x1, y1 = cvrp_instance.cities[route[i]]
        x2, y2 = cvrp_instance.cities[route[i + 1]]
        plt.plot([x1, x2], [y1, y2], 'g-')
    
    plt.title("CVRP Optimal Route Visualization")
    plt.legend()
    plt.show()

def main():
    # Define CVRP parameters
    cities = [(2, 3), (5, 8), (1, 9), (7, 3)]  # City coordinates, excluding depot
    demands = [0, 2, 3, 4, 5]  # Demand for each city
    capacity = 10  # Vehicle capacity
    depot_index = 0
    cvrp_instance = CVRP(cities=[(0, 0)] + cities, demands=[0]+demands, capacity=capacity, depot_index=depot_index)

    # Initialize State, Action, and Policy
    initial_state = State(cvrp_instance)
    action_generator = Action(cvrp_instance, initial_state)
    
    # Value network setup
    input_dim = len(initial_state.encode_state())
    value_network = ValueNetwork(input_size=input_dim, hidden_dim=16)
    optimizer = torch.optim.Adam(value_network.parameters(), lr=0.001)
    
    # Initialize Policy
    policy = Policy(cvrp_instance, initial_state, value_network, gamma=0.9)
    
    # Run policy iteration
    optimal_policy = policy_iteration(policy, value_network, optimizer, num_iterations=10)
    
    # Construct the optimal route from the policy
    current_state = initial_state
    route = [cvrp_instance.depot_index]  # Start from the depot

    while not current_state.is_terminal():
        action = optimal_policy.select_action(current_state)
        route.extend(action[1:-1])  # Exclude the depot from intermediate actions in the route
        for city in action[1:-1]:  # Transition the state to mark cities visited
            current_state.transition(city)
    
    # Append depot at the end of the route
    route.append(cvrp_instance.depot_index)

    print("Optimal Route:", route)
    visualize_route(cvrp_instance, route)

if __name__ == "__main__":
    main()
