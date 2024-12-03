import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from utils import solve_milp_with_value_function
from pyscipopt import Model

class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(ValueNetwork, self).__init__()
        if hidden_dim == 0:
            self.fc1 = nn.Linear(input_size, 1)
        else:
            self.fc1 = nn.Linear(input_size, hidden_dim*4)
            self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
            self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        
        for layer in [layer for layer in self.children() if isinstance(layer, nn.Linear)] :
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        if self.hidden_dim == 0:
            return self.fc1(x)
        else :
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
    
    def train_value_network(self, optimizer, num_episodes=1000, lower_bounds=None, batch_size=32):
        """
        Train value network using data from policy evaluation and Monte Carlo returns.

        Args:
            optimizer: Optimizer for the value network.
            num_episodes: Number of episodes to sample.
            lower_bounds: Optional list of lower bounds for returns.
            batch_size: Batch size for training.
        """
        self.value_net.train()
        all_losses = []

        for episode in range(num_episodes):
            # Generate an episode using the current policy
            states, actions, rewards = self.generate_episode()

            # Compute discounted returns
            returns = self.compute_returns(rewards)

            # Apply lower bounds if provided
            if lower_bounds:
                returns = [max(ret, lb) for ret, lb in zip(returns, lower_bounds)]

            # Prepare tensors
            state_tensors = torch.FloatTensor([s.encode_state() for s in states])
            returns_tensor = torch.FloatTensor(returns).unsqueeze(1)

            # Batch processing
            for i in range(0, len(states), batch_size):
                batch_states = state_tensors[i:i + batch_size]
                batch_returns = returns_tensor[i:i + batch_size]

                # Forward pass
                optimizer.zero_grad()
                value_predictions = self.value_net(batch_states)

                # Compute loss
                loss = F.mse_loss(value_predictions, batch_returns)

                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

                all_losses.append(loss.item())

            # Logging every 10 episodes
            if episode % 10 == 0:
                avg_loss = np.mean(all_losses[-10:])
                print(f"Episode {episode}, Avg Loss: {avg_loss:.4f}")

        print("Value network training complete.")

    
    def policy_evaluation(self, optimizer, gamma=0.9):
        """Evaluate current policy"""
        total_cost, next_state = self.policy_evaluation_with_milp(
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
    def policy_evaluation_with_milp(self, cvrp_instance, state, value_net):
        """
        Evaluates the current policy using MILP with SCIP and value function approximation.

        Args:
            cvrp_instance: CVRP instance with cities, demands, and capacity.
            state: Current state of the CVRP (unvisited cities, demands, etc.).
            value_net: Neural network for estimating the value function.

        Returns:
            total_cost: Cost of the solution found by MILP.
            next_state: Updated state after applying the action.
        """
        model = Model("CVRP_with_Value_Function")
        model.setParam("limits/time", 900)  # Limit to 900 seconds (15 minutes)
        #model.setParam("heuristics/aggressiveness", 1)
        model.setParam("limits/nodes", 100000)
        model.setParam("limits/gap", 0.01)  # Stop when the gap is below 1%
        n = cvrp_instance.num_cities
        depot = 0  # Depot index

        # Decision variables
        x = {}
        for v in range(cvrp_instance.num_vehicles):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x[v, i, j] = model.addVar(vtype="B", name=f"x_{v}_{i}_{j}")

        # Subtour elimination variables
        u = {}
        for i in range(n):
            u[i] = model.addVar(lb=0, ub=n - 1, vtype="C", name=f"u_{i}")

        # Objective: Immediate cost + future value
        immediate_cost = sum(
            cvrp_instance.distance_matrix[i][j] * x[v, i, j]
            for v in range(cvrp_instance.num_vehicles)
            for i in range(n)
            for j in range(n)
            if i != j
        )

        # Estimate future value using the value network
        state_encoding = torch.FloatTensor(state.encode_state()).unsqueeze(0)
        with torch.no_grad():
            future_value = value_net(state_encoding).item()

        # Set the objective
        model.setObjective(immediate_cost + self.gamma * future_value, "minimize")

        # Constraints
        # 1. Each city must be visited exactly once
        for i in range(1, n):
            model.addCons(
                sum(x[v, i, j] for v in range(cvrp_instance.num_vehicles) for j in range(n) if j != i) == 1
            )

        # 2. Flow conservation for each vehicle
        for v in range(cvrp_instance.num_vehicles):
            for i in range(1, n):
                model.addCons(
                    sum(x[v, i, j] for j in range(n) if j != i) ==
                    sum(x[v, j, i] for j in range(n) if j != i)
                )

        # 3. Depot constraints: Each vehicle starts and ends at the depot
        for v in range(cvrp_instance.num_vehicles):
            model.addCons(sum(x[v, depot, j] for j in range(1, n)) == 1)
            model.addCons(sum(x[v, i, depot] for i in range(1, n)) == 1)

        # 4. Capacity constraints
        for v in range(cvrp_instance.num_vehicles):
            model.addCons(
                sum(cvrp_instance.demands[i] * x[v, i, j] for i in range(1, n) for j in range(n) if i != j) <= cvrp_instance.capacity
            )

        # 5. MTZ subtour elimination
        for v in range(cvrp_instance.num_vehicles):
            for i in range(1, n):
                for j in range(1, n):
                    if i != j:
                        model.addCons(u[i] - u[j] + n * x[v, i, j] <= n - 1)

        # Solve the model
        model.optimize()

        # Extract results
        if model.getStatus() == "optimal":
            total_cost = model.getObjVal()

            # Extract solution to determine next state
            visited_cities = set()
            for v in range(cvrp_instance.num_vehicles):
                for i in range(n):
                    for j in range(n):
                        if i != j and model.getVal(x[v, i, j]) > 0.5:
                            visited_cities.add(j)

            # Update the next state based on visited cities
            next_state = state.transition(list(visited_cities))
            return total_cost, next_state

        print("MILP did not find an optimal solution.")
        return float('inf'), state



    def solve_with_pure_milp(self, cvrp_instance, state):
        """
        Solves CVRP using pure MILP without value function approximation
        """
        model = Model("CVRP_Pure_MILP")
        model.setParam("limits/time", 300)  # 5 minute limit
        n = cvrp_instance.num_cities
        depot = 0

        # Decision variables
        x = {}
        for v in range(cvrp_instance.num_vehicles):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x[v, i, j] = model.addVar(vtype="B", name=f"x_{v}_{i}_{j}")

        # Subtour elimination variables
        u = {}
        for i in range(n):
            u[i] = model.addVar(lb=0, ub=n - 1, vtype="C", name=f"u_{i}")

        # Objective: Only immediate cost
        objective = sum(
            cvrp_instance.distance_matrix[i][j] * x[v, i, j]
            for v in range(cvrp_instance.num_vehicles)
            for i in range(n)
            for j in range(n)
            if i != j
        )

        model.setObjective(objective, "minimize")

        # Add standard CVRP constraints
        # Each city visited once
        for i in range(1, n):
            model.addCons(
                sum(x[v, i, j] for v in range(cvrp_instance.num_vehicles) for j in range(n) if j != i) == 1
            )

        # Flow conservation
        for v in range(cvrp_instance.num_vehicles):
            for i in range(1, n):
                model.addCons(
                    sum(x[v, i, j] for j in range(n) if j != i) ==
                    sum(x[v, j, i] for j in range(n) if j != i)
                )

        # Depot constraints
        for v in range(cvrp_instance.num_vehicles):
            model.addCons(sum(x[v, depot, j] for j in range(1, n)) == 1)
            model.addCons(sum(x[v, i, depot] for i in range(1, n)) == 1)

        # Capacity constraints
        for v in range(cvrp_instance.num_vehicles):
            model.addCons(
                sum(cvrp_instance.demands[i] * x[v, i, j] 
                    for i in range(1, n) 
                    for j in range(n) if i != j) <= cvrp_instance.capacity
            )

        # MTZ subtour elimination
        for v in range(cvrp_instance.num_vehicles):
            for i in range(1, n):
                for j in range(1, n):
                    if i != j:
                        model.addCons(u[i] - u[j] + n * x[v, i, j] <= n - 1)

        # Solve and return results
        model.optimize()
        
        if model.getStatus() == "optimal":
            return model.getObjVal(), x
        else:
            print(f"Solver status: {model.getStatus()}")
            return None, None
