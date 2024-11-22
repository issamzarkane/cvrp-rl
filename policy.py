import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import solve_milp_with_value_function

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class Policy:
    def __init__(self, cvrp_instance, initial_state, value_net, gamma=0.9):
        self.cvrp = cvrp_instance
        self.initial_state = initial_state
        self.value_net = value_net
        self.gamma = gamma
    
    def select_action(self, state):
        """Select best action using current value network"""
        self.value_net.eval()
        best_action = None
        min_total_cost = float('inf')
        
        for action in self.cvrp.get_feasible_actions(state):
            next_state = state.transition(action)
            immediate_cost = self.compute_cost(action)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(next_state.encode_state()).unsqueeze(0)
                future_cost = self.value_net(state_tensor).squeeze().item()
                
            total_cost = immediate_cost + self.gamma * future_cost
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_action = action
                
        return best_action
    
    def compute_cost(self, action):
        """Compute total distance cost of a route"""
        total_cost = 0
        for i in range(len(action) - 1):
            city1, city2 = action[i], action[i + 1]
            total_cost += self.cvrp.distance_matrix[city1][city2]
        return total_cost
    
    def estimate_value(self, state):
        """Estimate value of a state using value network"""
        state_tensor = torch.FloatTensor(state.encode_state()).unsqueeze(0)
        with torch.no_grad():
            value = self.value_net(state_tensor).squeeze().item()
        return value
    
    def generate_episode(self):
        """Generate episode using current policy"""
        state = self.initial_state
        states, actions, rewards = [], [], []
        
        while not state.is_terminal():
            action = self.select_action(state)
            next_state = state.transition(action)
            reward = -self.compute_cost(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            
        return states, actions, rewards
    
    def compute_returns(self, rewards):
        """Compute discounted returns"""
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def train_value_network(self, optimizer, num_episodes=1000):
        """Train value network using Monte Carlo sampling"""
        self.value_net.train()
        
        for episode in range(num_episodes):
            states, actions, rewards = self.generate_episode()
            returns = self.compute_returns(rewards)
            
            state_tensors = torch.FloatTensor([s.encode_state() for s in states])
            returns_tensor = torch.FloatTensor(returns).unsqueeze(1)
            
            optimizer.zero_grad()
            value_predictions = self.value_net(state_tensors)
            loss = F.mse_loss(value_predictions, returns_tensor)
            loss.backward()
            optimizer.step()
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Loss: {loss.item():.4f}")
    
    def policy_evaluation(self, optimizer, gamma=0.9):
        """Evaluate current policy"""
        total_cost, next_state = solve_milp_with_value_function(
            self.cvrp, self.initial_state, self.value_net)
        return total_cost, next_state
    
    def policy_improvement(self):
        """Improve policy based on current value function"""
        state = self.initial_state
        best_action = self.select_action(state)
        return best_action
    
    def policy_iteration(self, optimizer, num_iterations=10):
        """Main policy iteration loop"""
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Policy Evaluation
            total_cost, next_state = self.policy_evaluation(optimizer)
            
            # Policy Improvement
            best_action = self.policy_improvement()
            
            # Train Value Network
            self.train_value_network(optimizer)
            
            print(f"Total Cost: {total_cost}")