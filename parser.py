import os
import numpy as np


def parse_cvrplib_from_folders(folder_paths):
    """
    Parses all CVRPLIB `.vrp` files from the specified folders.

    Parameters:
        folder_paths (list of str): List of folder paths (e.g., ["path/to/A", "path/to/B"]).

    Returns:
        dict: A dictionary where keys are filenames (without extensions), and values are tuples:
              (depot, cities, demands, capacity).
    """
    parsed_instances = {}
    for folder in folder_paths:
        for filename in os.listdir(folder):
            if filename.endswith(".vrp"):
                filepath = os.path.join(folder, filename)
                instance_name = os.path.splitext(filename)[0]  # Remove file extension
                parsed_instances[instance_name] = parse_cvrplib(filepath)

    return parsed_instances


def parse_cvrplib(filepath):
    """
    Parses a single CVRPLIB file.

    Parameters:
        filepath (str): Path to the CVRPLIB instance file.

    Returns:
        tuple: (depot, cities, demands, capacity, num_vehicles).
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    section = None
    cities = []
    demands = []
    depot = None
    capacity = 0
    num_vehicles = int(os.path.basename(filepath).split('-k')[1].split('.')[0])  # Extract k value from filename

    for line in lines:
        if "NODE_COORD_SECTION" in line:
            section = "coordinates"
        elif "DEMAND_SECTION" in line:
            section = "demands"
        elif "DEPOT_SECTION" in line:
            section = "depot"
        elif section == "coordinates" and "EOF" not in line and len(line.split()) == 3:
            _, x, y = map(float, line.split())
            cities.append((x, y))
        elif section == "demands" and "EOF" not in line and len(line.split()) == 2:
            _, demand = map(int, line.split())
            demands.append(demand)
        elif section == "depot" and "EOF" not in line and len(line.split()) == 1:
            depot = int(line.split()[0]) - 1  # Adjust for 0-indexing
        elif "CAPACITY" in line:
            capacity = int(line.split()[-1])

    return depot, cities, demands, capacity, num_vehicles

def parse_solution_from_folders(folder_paths):
    """
    Parses all `.sol` files from specified folders.

    Parameters:
        folder_paths (list of str): List of folder paths (e.g., ["path/to/A", "path/to/B"]).

    Returns:
        dict: A dictionary where keys are filenames (without extensions), and values are tuples:
              (routes, cost) representing the optimal solution.
    """
    parsed_solutions = {}
    
    # Loop over each folder
    for folder in folder_paths:
        for filename in os.listdir(folder):
            if filename.endswith(".sol"):
                filepath = os.path.join(folder, filename)
                instance_name = os.path.splitext(filename)[0]  # Remove file extension
                parsed_solutions[instance_name] = parse_solution(filepath)
    
    return parsed_solutions


def parse_solution(filepath):
    """
    Parses a CVRPLIB `.sol` file to extract routes and the total cost.

    Parameters:
        filepath (str): Path to the `.sol` file.

    Returns:
        tuple: (routes, cost), where:
               - routes is a list of lists (e.g., [[0, 1, 3, 5, 0], [0, 2, 4, 0]])
               - cost is the total cost (float or int).
    """
    routes = []
    cost = None

    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("Route"):
                # Extract route numbers
                route = list(map(int, line.split(":")[1].strip().split()))
                routes.append(route)
            elif line.startswith("Cost"):
                # Extract total cost
                cost = float(line.split()[1].strip())

    return routes, cost

