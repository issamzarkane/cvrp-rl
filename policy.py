import torch
import torch.nn as nn
import torch.optim as optim
import random

class Policy:
    def __init__(self, cvrp_instance, state, value_net, gamma=0.9):
        self.cvrp = cvrp_instance
        self.state = state
        self.value_net = value_net
        self.gamma = gamma  # Decay factor for policy iteration

    def select_action(self, state):
        # Select the best action based on current value network estimate
        best_action = None
        min_cost = float('inf')
        
        for action in self.cvrp.get_feasible_actions(state):
            next_state = self.cvrp.transition(state, action)
            action_cost = self.cvrp.cost(action)
            estimated_value = self.value_net(torch.FloatTensor(next_state)).item()
            total_cost = action_cost + estimated_value
            
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
    # Update policy to minimize the cost for each state
    new_policy_actions = {}
    for state in policy.cvrp.get_all_states():
        best_action = policy.select_action(state)
        new_policy_actions[state] = best_action
    
    policy.actions = new_policy_actions  # Replace policy's actions with improved actions

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