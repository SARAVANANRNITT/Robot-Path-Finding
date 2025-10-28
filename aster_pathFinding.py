import heapq
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

# ------------------------------
# 1. Define the Heuristics
# ------------------------------

def manhattan(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    """Euclidean distance heuristic."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def chebyshev(a, b):
    """Chebyshev distance heuristic."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

HEURISTICS = {
    "Manhattan": manhattan,
    "Euclidean": euclidean,
    "Chebyshev": chebyshev
}


# ------------------------------
# 2. A* Algorithm Implementation
# ------------------------------

def neighbors(cell, grid):
    """Get valid 4-directional neighbors."""
    x, y = cell
    rows, cols = grid.shape
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
            yield (nx, ny)

def astar(grid, start, goal, heuristic_func):
    """A* pathfinding algorithm."""
    start_time = time.time()
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    g = {start: 0}
    f = {start: heuristic_func(start, goal)}
    
    closed = set()
    nodes_expanded = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        nodes_expanded += 1

        if current == goal:
            elapsed = time.time() - start_time
            return True, g[current], nodes_expanded, elapsed
        
        closed.add(current)
        
        for nb in neighbors(current, grid):
            tentative_g = g[current] + 1
            if nb in closed:
                continue
            if nb not in g or tentative_g < g[nb]:
                g[nb] = tentative_g
                f[nb] = tentative_g + heuristic_func(nb, goal)
                heapq.heappush(open_set, (f[nb], nb))
                
    elapsed = time.time() - start_time
    return False, None, nodes_expanded, elapsed


# ------------------------------
# 3. Random Grid Generator
# ------------------------------

def generate_grid(rows, cols, obstacle_prob=0.2):
    """Generate a random grid with obstacles."""
    grid = (np.random.rand(rows, cols) < obstacle_prob).astype(int)
    return grid

def random_free_cell(grid):
    """Pick a random free cell (value 0)."""
    rows, cols = grid.shape
    while True:
        x, y = random.randint(0, rows-1), random.randint(0, cols-1)
        if grid[x, y] == 0:
            return (x, y)

def bfs_reachable(grid, start, goal):
    """Check if start and goal are connected (reachable)."""
    q = deque([start])
    visited = {start}
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return False


# ------------------------------
# 4. Experiment Runner
# ------------------------------

def run_experiments(n_trials=20, rows=25, cols=25, obstacle_prob=0.25):
    results = []
    for t in range(n_trials):
        grid = generate_grid(rows, cols, obstacle_prob)
        start = random_free_cell(grid)
        goal = random_free_cell(grid)
        
        # Ensure start != goal and reachable
        attempts = 0
        while (goal == start or not bfs_reachable(grid, start, goal)) and attempts < 10:
            grid = generate_grid(rows, cols, obstacle_prob)
            start = random_free_cell(grid)
            goal = random_free_cell(grid)
            attempts += 1
        
        for name, func in HEURISTICS.items():
            found, path_len, nodes_expanded, elapsed = astar(grid, start, goal, func)
            results.append({
                "Trial": t+1,
                "Heuristic": name,
                "Found": found,
                "Path Length": path_len if found else None,
                "Nodes Expanded": nodes_expanded,
                "Time (s)": elapsed
            })
    return pd.DataFrame(results)


# ------------------------------
# 5. Run and Compare Performance
# ------------------------------

df = run_experiments(n_trials=30, rows=30, cols=30, obstacle_prob=0.25)

summary = df.groupby("Heuristic").agg({
    "Found": "mean",
    "Path Length": "mean",
    "Nodes Expanded": "mean",
    "Time (s)": "mean"
}).reset_index()

print("\n=== A* Heuristic Comparison Summary ===\n")
print(summary)

# ------------------------------
# 6. Plot Performance Comparison
# ------------------------------

plt.figure(figsize=(8,5))
plt.bar(summary["Heuristic"], summary["Nodes Expanded"], color='skyblue')
plt.title("Average Nodes Expanded by Heuristic")
plt.ylabel("Average Nodes Expanded")
plt.xlabel("Heuristic")
plt.show()

plt.figure(figsize=(8,5))
plt.bar(summary["Heuristic"], summary["Time (s)"], color='lightgreen')
plt.title("Average Runtime by Heuristic")
plt.ylabel("Average Time (seconds)")
plt.xlabel("Heuristic")
plt.show()
