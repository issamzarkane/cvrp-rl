import numpy as np
import torch
import wandb
from cvrp import CVRP
from state import State
from action import Action
from parser import parse_cvrplib_from_folders, parse_solution_from_folders
from policy import Policy, ValueNetwork, policy_evaluation_with_milp, policy_improvement
import matplotlib.pyplot as plt

# Assuming the parse_solution function is already defined to read .sol files and extract routes and costs
def compute_gap(model_cost, optimal_cost):
    """
    Computes the gap between the model's total cost and the optimal cost.
    Gap is calculated as a percentage difference.
    """
    return abs(model_cost - optimal_cost) / optimal_cost * 100

def main():
    # Paths to your CVRPLIB dataset folders
    folder_paths = [
        "C:/Users/issam/OneDrive/Desktop/phd/Vrp-Set-A/A", 
        "C:/Users/issam/OneDrive/Desktop/phd/Vrp-Set-B/B"
    ]

    # Parse all CVRPLIB instances from the folders (returns a dictionary of (depot, cities, demands, capacity) tuples)
    instances = parse_cvrplib_from_folders(folder_paths)

    # Parse all optimal solutions from the corresponding .sol files
    solutions = parse_solution_from_folders(folder_paths)

    # Initialize a dictionary to store gaps for each instance
    all_gaps = {}

    # Iterate through each instance in the parsed data
    for instance_name, (depot, cities, demands, capacity) in instances.items():
        print(f"Processing instance: {instance_name}")

        # Retrieve the optimal cost for the current instance from the solutions dictionary
        routes, optimal_cost = solutions.get(instance_name, (None, None))
        
        if optimal_cost is None:
            print(f"Warning: No solution found for {instance_name}. Skipping.")
            continue

        # Create CVRP instance for the current dataset
        cvrp_instance = CVRP(cities=[cities[depot]] + cities, demands=[0] + demands, capacity=capacity, depot_index=0)

        # Initialize State, Action, and Policy for policy iteration
        initial_state = State(cvrp_instance)
        action_generator = Action(cvrp_instance, initial_state)
        input_dim = len(initial_state.encode_state())
        value_network = ValueNetwork(input_size=input_dim, hidden_dim=16)
        optimizer = torch.optim.Adam(value_network.parameters(), lr=0.001)
        policy = Policy(cvrp_instance, initial_state, value_network, gamma=0.9)

        # Track the gap for the current instance
        gaps = []

        # Run Policy Iteration and track the gap for this instance
        for iteration in range(10):  # Adjust the number of iterations as needed
            print(f"Iteration {iteration + 1}")

            # Perform Policy Evaluation with MILP
            total_cost, next_state = policy_evaluation_with_milp(cvrp_instance, initial_state, value_network, optimizer)

            # Compute the gap between the model's solution and the optimal solution
            gap = compute_gap(total_cost, optimal_cost)
            gaps.append(gap)

            print(f"Model total cost: {total_cost}, Optimal total cost: {optimal_cost}, Gap: {gap}%")

        # Store the gap values for this instance
        all_gaps[instance_name] = gaps

    # Plot gaps for each instance
    plt.figure(figsize=(12, 8))

    for instance_name, gaps in all_gaps.items():
        plt.plot(range(1, len(gaps) + 1), gaps, marker='o', label=f"{instance_name} Gap")

    plt.xlabel('Iteration')
    plt.ylabel('Gap (%)')
    plt.title('Gap Between Model Solution and Optimal Solution Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
