import torch
import torch.nn as nn
import torch.optim as optim
import random
from pyscipopt import Model
import torch
from utils import solve_milp_with_value_function
from utils import transition, cost
from action import Action

class Policy:
    def __init__(self, cvrp_instance, state, value_net, gamma=0.9):
        self.cvrp = cvrp_instance
        self.state = state
        self.value_net = value_net
        self.gamma = gamma  # Decay factor for policy iteration

    def select_action(self, state):
        """
        Selects the best action based on the current value network estimate.

        Parameters:
            state (State): The current state.

        Returns:
            tuple: The best action (route) minimizing the total cost.
        """
        # Initialize variables for tracking the best action
        best_action = None
        min_cost = float('inf')

        # Create an Action object to generate all feasible actions
        action_generator = Action(self.cvrp, state)

        for action in action_generator.get_all_actions():
            # Use the transition function from utils.py to get the next state
            next_state = transition(state, action)

            if next_state is None:  # Skip invalid actions
                continue

            # Compute the immediate cost of the action
            action_cost = cost(self.cvrp, action)

            # Estimate the cost-to-go using the value network
            state_tensor = torch.FloatTensor(next_state.encode_state())
            estimated_value = self.value_net(state_tensor).item()

            # Calculate the total cost for the action
            total_cost = action_cost + estimated_value

            # Update the best action if a lower-cost action is found
            if total_cost < min_cost:
                min_cost = total_cost
                best_action = action

        return best_action


class ValueNetwork(nn.Module):
    def __init__(self, input_size,hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output is a single value

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def policy_evaluation(data, value_net, optimizer=optim, gamma=0.9):
    mse_loss = nn.MSELoss()
    all_loss = 0.0
    for k_prime, (states, costs) in enumerate(data):  # data holds (s, c) pairs from each iteration
        decay_factor = gamma ** k_prime
        # Randomly sample validation set
        train_data = random.sample(states, int(0.8 * len(states)))
        val_data = [s for s in states if s not in train_data]

        # Training
        value_net.train()
        for state, cost in train_data:
            optimizer.zero_grad()
            predicted_value = value_net(torch.FloatTensor(state))
            loss = mse_loss(predicted_value, torch.FloatTensor([cost])) * decay_factor
            loss.backward()
            optimizer.step()
            all_loss += loss.item()

        # Validation
        value_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for state, cost in val_data:
                predicted_value = value_net(torch.FloatTensor(state))
                val_loss += mse_loss(predicted_value, torch.FloatTensor([cost])).item()

    print(f"Training Loss: {all_loss}, Validation Loss: {val_loss}")

def policy_iteration(policy, value_net, optimizer=optim, num_iterations=10):
    for k in range(num_iterations):
        print(f"Iteration {k+1}/{num_iterations}")
        
        # Step 1: Policy Evaluation
        data = gather_evaluation_data(policy, value_net)
        policy_evaluation(data, value_net, optimizer, gamma=policy.gamma)
        
        # Step 2: Policy Improvement
        policy_improvement(policy, value_net)

def policy_improvement(policy, value_net):
    state = policy.state
    best_action = policy.select_action(state)
    # Update policy to minimize the cost for each state
    #new_policy_actions = {}
    #for state in policy.cvrp.get_all_states():
     #   best_action = policy.select_action(state)
      #  new_policy_actions[state] = best_action
    
    #policy.actions = new_policy_actions  # Replace policy's actions with improved actions

def gather_evaluation_data(policy, value_net, num_samples=100):
    # Collect data of state-action pairs and cumulative costs
    data = []
    for _ in range(num_samples):
        state = policy.cvrp.sample_random_start_state()
        sample_path = []
        total_cost = 0
        while not policy.cvrp.is_terminal(state):
            action = policy.select_action(state)
            next_state = policy.cvrp.transition(state, action)
            cost = policy.cvrp.cost(action)
            sample_path.append((state, cost))
            total_cost += cost
            state = next_state
        data.append(sample_path)
    return data


def policy_evaluation_with_milp(cvrp_instance, state, value_net, optimizer, gamma=0.9):
    """
    Policy evaluation using MILP for immediate cost and cost-to-go.
    """
    route = solve_milp_with_value_function(cvrp_instance, state, value_net)
    print(route)
    total_cost = sum(cvrp_instance.distance_matrix[i][j] for i, j in route)

    for _, j in route:
        if j != cvrp_instance.depot_index:
            state.transition(j)

    state_tensor = torch.FloatTensor(state.encode_state())
    cost_to_go = value_net(state_tensor).item()

    return total_cost + cost_to_go, state


