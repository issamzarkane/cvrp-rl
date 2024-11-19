# cvrp-rl
Implementation for [Reinforcement Learning with Combinatorial Actions: An Application to Vehicle Routing](https://doi.org/10.48550/arXiv.2010.12001) <br>
## CVRP :
CVRP instance : 
- $n-1$ cities : $(0,1,...,n-1)$ where $0$ is the _depot_. 
- each city besides the depot have a demand $d_i$.
- distance between cities is denoted $\Delta_{ij}$.
- vehicle with capacity $Q$. 

__Goal__ : Produce routes that start and end at the depot such that each city $i>0$ is visited exactly once, total demand for each route shouldn't exceed $Q$, and the total distance of all routes should be minimized. (
_We assume the number of routes we can serve is unbounded_)

## MDP :
### State Space :
State $s$ : set of as-yet visited cities <br>
States are represented using a binary encoding :
- $0$ if the city was visited.
- $1$ if not.<br>

In this case $S = \{ 0,1 \}^n $, by convention we mark the depot visited.

### Action Space :
Action $a$ : feasible route starting and ending at the depot and covering at least one city. <br>
Action space $A$  corresponds to the set of all partial permutations of $n-1$ cities.
### Transition 