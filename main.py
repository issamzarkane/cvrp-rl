import numpy as np
import torch
import wandb
from cvrp import CVRP
from state import State
from action import Action
from parser import parse_cvrplib_from_folders, parse_solution_from_folders
from policy import Policy, ValueNetwork
import matplotlib.pyplot as plt
from ORtools import ORToolsSolver

# Assuming the parse_solution function is already defined to read .sol files and extract routes and costs
def compute_gap(model_cost, optimal_cost):
    """
    Computes the gap between the model's total cost and the optimal cost.
    Gap is calculated as a percentage difference.
    """
    return abs(model_cost - optimal_cost) / optimal_cost * 100

def main(): 
            folder_paths = [
                "C:/Users/issam/OneDrive/Desktop/phd/Vrp-Set-A/A", 
                "C:/Users/issam/OneDrive/Desktop/phd/Vrp-Set-B/B"
            ]

            # Parse instances only (remove solutions parsing)
            instances = parse_cvrplib_from_folders(folder_paths)
            all_gaps = {}

            for instance_name, (depot, cities, demands, capacity) in instances.items():
                print(f"Processing instance: {instance_name}")

                cvrp_instance = CVRP(cities=[cities[depot]] + cities, demands=[0] + demands, capacity=capacity, depot_index=0)
        
                # Get OR-Tools solution
                ortools_solver = ORToolsSolver(cvrp_instance)
                ortools_cost = ortools_solver.solve()
        
                # Initialize components for policy iteration
                initial_state = State(cvrp_instance)
                input_dim = len(initial_state.encode_state())
                value_network = ValueNetwork(input_size=input_dim, hidden_dim=16)
                optimizer = torch.optim.Adam(value_network.parameters(), lr=0.001)
                policy = Policy(cvrp_instance, initial_state, value_network, gamma=0.9)

                gaps = []
                for iteration in range(10):
                    print(f"Iteration {iteration + 1}")
                    total_cost, next_state = policy.policy_evaluation_with_milp(cvrp_instance, initial_state, value_network, optimizer)
                    gap = compute_gap(total_cost, ortools_cost)
                    gaps.append(gap)
                    print(f"Model cost: {total_cost}, OR-Tools cost: {ortools_cost}, Gap: {gap}%")

                all_gaps[instance_name] = gaps

                # After collecting all gaps, plot the results
                plt.figure(figsize=(12, 8))
    
                # Plot individual instance gaps
                for instance_name, gaps in all_gaps.items():
                    plt.plot(range(1, len(gaps) + 1), gaps, marker='o', label=f"{instance_name}")
    
                # Calculate and plot average gap across all instances
                avg_gaps = np.mean([gaps for gaps in all_gaps.values()], axis=0)
                plt.plot(range(1, len(avg_gaps) + 1), avg_gaps, 'r--', linewidth=2, label='Average Gap')
    
                plt.xlabel('Iteration')
                plt.ylabel('Gap to OR-Tools Solution (%)')
                plt.title('Model Performance vs OR-Tools Over Iterations')
                plt.legend()
                plt.grid(True)
                plt.show()

if __name__ == "__main__":
    main()

