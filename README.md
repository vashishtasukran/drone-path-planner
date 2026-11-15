Interactive 3D PyBullet sandbox where two simulated drones plan A* routes around walls you edit with clicks. Live metrics show path length, cells explored and replan time.

## Run it
1. Install PyBullet: `conda install -c conda-forge pybullet` (or `pip install pybullet`)
2. Put `planner.py` and `main-sim.py` in the same folder
3. Run: `python main-sim.py`

## Controls
- **MODE: ADD / REMOVE blocks** buttons in the side panel
- Click cells on the green grid to edit the wall
- **edit layer (x)** slider changes which layer you edit
- **REPLAN + RESTART** restarts both drones

## Status
TKS Iterate v1. Drone 2 avoids drone 1's whole route, so it is collision-free by design but limited.
v2 plan: noisy sensors, a simulated camera and physics (gravity, wind).
