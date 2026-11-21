"""
Iterate v2 simulation: a robot explores an UNKNOWN grid with a simulated
360-degree lidar and replans with D* Lite whenever it discovers a wall.

- Hidden walls start as faint grey boxes (the robot doesn't know them).
- When a lidar ray hits one, it turns red and gets added to D* Lite.
- The blue line is the current planned path.

Run:  python sim_dstar.py            (GUI)
      python sim_dstar.py --headless (no window, prints results)
      python sim_dstar.py --seed 5   (different random map)
"""

import argparse
import math
import random
import time
from collections import deque

import pybullet as p
import pybullet_data

from v2.dstar_lite import DStarLite, INF, MOVES

SIZE = 10
START, GOAL = (0, 0), (SIZE - 1, SIZE - 1)
WALL_DENSITY = 0.22
LIDAR_RAYS = 72          # one ray every 5 degrees
LIDAR_RANGE = 3.0        # in cells
LIDAR_HEIGHT = 0.3
STEP_FRAMES = 12         # animation frames per cell move


# ---------- map ----------
def reachable(blocked):
    """Quick BFS: is the goal reachable on the TRUE map?"""
    seen, q = {START}, deque([START])
    while q:
        c = q.popleft()
        if c == GOAL:
            return True
        for dx, dy in MOVES:
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE and n not in blocked and n not in seen:
                seen.add(n)
                q.append(n)
    return False


def make_map(rng):
    while True:
        blocked = {(x, y) for x in range(SIZE) for y in range(SIZE)
                   if rng.random() < WALL_DENSITY} - {START, GOAL}
        if reachable(blocked):
            return blocked


# ---------- world building ----------
def build_world(true_walls):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [SIZE / 2 - 0.5, SIZE / 2 - 0.5, 0])

    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5],
                                  rgbaColor=[0.6, 0.6, 0.6, 0.25])
    wall_ids = {}
    for (x, y) in true_walls:
        bid = p.createMultiBody(0, box_col, box_vis, [x, y, 0.5])
        wall_ids[bid] = (x, y)

    # Robot: a small cylinder.
    r_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.4,
                                rgbaColor=[0.1, 0.4, 0.9, 1])
    robot = p.createMultiBody(0, -1, r_vis, [START[0], START[1], 0.2])

    # Goal marker.
    g_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[0.1, 0.8, 0.2, 1])
    p.createMultiBody(0, -1, g_vis, [GOAL[0], GOAL[1], 0.25])

    # Grid lines on the floor.
    for i in range(SIZE + 1):
        v = i - 0.5
        p.addUserDebugLine([v, -0.5, 0.01], [v, SIZE - 0.5, 0.01], [0.3, 0.3, 0.3])
        p.addUserDebugLine([-0.5, v, 0.01], [SIZE - 0.5, v, 0.01], [0.3, 0.3, 0.3])

    p.resetDebugVisualizerCamera(cameraDistance=11, cameraYaw=0, cameraPitch=-70,
                                 cameraTargetPosition=[SIZE / 2 - 0.5, SIZE / 2 - 0.5, 0])
    return wall_ids, robot


# ---------- lidar ----------
def scan(pos, wall_ids, gui, ray_ids):
    """Cast LIDAR_RAYS rays around the robot. Return the wall cells they hit."""
    ox, oy = pos
    starts, ends = [], []
    for i in range(LIDAR_RAYS):
        a = 2 * math.pi * i / LIDAR_RAYS
        starts.append([ox, oy, LIDAR_HEIGHT])
        ends.append([ox + LIDAR_RANGE * math.cos(a), oy + LIDAR_RANGE * math.sin(a), LIDAR_HEIGHT])

    results = p.rayTestBatch(starts, ends)
    found = set()
    for i, (body, _, frac, hit_pos, _) in enumerate(results):
        hit = body in wall_ids
        if hit:
            found.add(wall_ids[body])
        if gui:
            end = hit_pos if hit else ends[i]
            color = [1, 0.2, 0.2] if hit else [0.3, 0.9, 0.3]
            ray_ids[i] = p.addUserDebugLine(starts[i], end, color, 1,
                                            replaceItemUniqueId=ray_ids[i])
    return found


# ---------- main loop ----------
def run(seed, gui):
    rng = random.Random(seed)
    true_walls = make_map(rng)

    p.connect(p.GUI if gui else p.DIRECT)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    wall_ids, robot = build_world(true_walls)
    body_of = {cell: bid for bid, cell in wall_ids.items()}

    # The robot starts knowing NOTHING about the walls.
    planner = DStarLite(SIZE, START, GOAL, blocked=set())
    planner.compute_shortest_path()

    cur = START
    steps = replans = 0
    ray_ids = [-1] * LIDAR_RAYS
    path_ids = []
    text_id = -1

    while cur != GOAL:
        # 1. Sense
        new_walls = scan(cur, wall_ids, gui, ray_ids) - planner.blocked

        # 2. Replan only if something new was found
        if new_walls:
            planner.move_robot(cur)
            planner.add_walls(new_walls)
            replans += 1
            for c in new_walls:
                p.changeVisualShape(body_of[c], -1, rgbaColor=[0.9, 0.15, 0.15, 1])

        if planner.get_g(cur) == INF:
            print("No path to goal.")
            break

        # 3. Draw the current plan and stats
        if gui:
            for lid in path_ids:
                p.removeUserDebugItem(lid)
            path = planner.path() or []
            path_ids = [p.addUserDebugLine([a[0], a[1], 0.05], [b[0], b[1], 0.05],
                                           [0.2, 0.5, 1], 4)
                        for a, b in zip(path, path[1:])]
            info = (f"steps {steps}   walls found {len(planner.blocked)}   "
                    f"replans {replans}   cells expanded {planner.expansions}")
            text_id = p.addUserDebugText(info, [-0.5, SIZE + 0.3, 0], [1, 1, 1], 1.3,
                                         replaceItemUniqueId=text_id)

        # 4. Move one cell
        nxt = planner.next_step(cur)
        if gui:
            for f in range(1, STEP_FRAMES + 1):
                t = f / STEP_FRAMES
                x = cur[0] + (nxt[0] - cur[0]) * t
                y = cur[1] + (nxt[1] - cur[1]) * t
                p.resetBasePositionAndOrientation(robot, [x, y, 0.2], [0, 0, 0, 1])
                time.sleep(1 / 60)
        else:
            p.resetBasePositionAndOrientation(robot, [nxt[0], nxt[1], 0.2], [0, 0, 0, 1])
        cur = nxt
        planner.move_robot(cur)
        steps += 1

    result = {"seed": seed, "reached": cur == GOAL, "steps": steps,
              "walls_found": len(planner.blocked), "true_walls": len(true_walls),
              "replans": replans, "expansions": planner.expansions}
    print(result)

    if gui:
        print("Done. Close the window to exit.")
        while p.isConnected():
            time.sleep(0.1)
    if p.isConnected():
        p.disconnect()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    run(args.seed, gui=not args.headless)