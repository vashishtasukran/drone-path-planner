"""
Measurement for Iterate v2: D* Lite vs a FAIR A* baseline on unknown maps.

Both robots start knowing nothing, use the same sensor (sees walls within
SENSOR_RANGE cells, line of sight), and take the same kind of moves.

- D* Lite: repairs its plan incrementally when walls are discovered.
- Fair A*: keeps its current path and replans FROM SCRATCH only when a newly
  discovered wall blocks that path (no pointless replans).

We count "cells expanded" = cells popped from the priority queue, summed over
the whole trip. Run with: python measure.py
"""

import heapq
import random
import statistics
from collections import deque

from v2.dstar_lite import DStarLite, MOVES

SENSOR_RANGE = 3
WALL_DENSITY = 0.22
MAPS_PER_SIZE = 50
SIZES = [10, 20, 30]


# ---------- map + sensor ----------
def reachable(size, start, goal, blocked):
    seen, q = {start}, deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            return True
        for dx, dy in MOVES:
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < size and 0 <= n[1] < size and n not in blocked and n not in seen:
                seen.add(n)
                q.append(n)
    return False


def make_map(size, rng):
    start, goal = (0, 0), (size - 1, size - 1)
    while True:
        blocked = {(x, y) for x in range(size) for y in range(size)
                   if rng.random() < WALL_DENSITY} - {start, goal}
        if reachable(size, start, goal, blocked):
            return start, goal, blocked


def line(a, b):
    """Grid cells on the line from a to b (Bresenham)."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err, cells = dx + dy, []
    while True:
        cells.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def sense(pos, size, true_walls):
    """Walls within range that the robot has line of sight to."""
    found = set()
    for x in range(pos[0] - SENSOR_RANGE, pos[0] + SENSOR_RANGE + 1):
        for y in range(pos[1] - SENSOR_RANGE, pos[1] + SENSOR_RANGE + 1):
            if not (0 <= x < size and 0 <= y < size) or (x, y) not in true_walls:
                continue
            if (x - pos[0]) ** 2 + (y - pos[1]) ** 2 > SENSOR_RANGE ** 2:
                continue
            # visible if no other wall sits between robot and this cell
            if all(c not in true_walls for c in line(pos, (x, y))[1:-1]):
                found.add((x, y))
    return found


# ---------- A* (forward, from scratch) ----------
def astar(size, start, goal, blocked):
    def h(c):
        return abs(c[0] - goal[0]) + abs(c[1] - goal[1])
    heap, g, parent, closed, expanded = [(h(start), 0, start)], {start: 0}, {}, set(), 0
    while heap:
        _, cost, c = heapq.heappop(heap)
        if c in closed:
            continue
        closed.add(c)
        expanded += 1
        if c == goal:
            path = [c]
            while c in parent:
                c = parent[c]
                path.append(c)
            return path[::-1], expanded
        for dx, dy in MOVES:
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < size and 0 <= n[1] < size and n not in blocked:
                if cost + 1 < g.get(n, float("inf")):
                    g[n], parent[n] = cost + 1, c
                    heapq.heappush(heap, (cost + 1 + h(n), cost + 1, n))
    return None, expanded


# ---------- the two robots ----------
def run_dstar(size, start, goal, true_walls):
    d = DStarLite(size, start, goal, blocked=set())
    d.compute_shortest_path()
    cur, steps, replans = start, 0, 0
    while cur != goal and steps < size * size * 4:
        new = sense(cur, size, true_walls) - d.blocked
        if new:
            d.move_robot(cur)
            d.add_walls(new)
            replans += 1
        cur = d.next_step(cur)
        d.move_robot(cur)
        steps += 1
    return d.expansions, steps, replans


def run_fair_astar(size, start, goal, true_walls):
    known = set()
    path, total = astar(size, start, goal, known)
    cur, i, steps, replans = start, 0, 0, 0
    while cur != goal and steps < size * size * 4:
        new = sense(cur, size, true_walls) - known
        known |= new
        # replan only if a new wall lands on the remaining path
        if new & set(path[i + 1:]):
            path, e = astar(size, cur, goal, known)
            total += e
            i = 0
            replans += 1
        i += 1
        cur = path[i]
        steps += 1
    return total, steps, replans


# ---------- experiment ----------
def main():
    rng = random.Random(42)
    print(f"{MAPS_PER_SIZE} random maps per size, {int(WALL_DENSITY*100)}% walls, "
          f"sensor range {SENSOR_RANGE}\n")
    print(f"{'grid':>6} | {'D* Lite expanded':>16} | {'fair A* expanded':>16} | "
          f"{'reduction':>9} | {'D* Lite wins':>12}")
    print("-" * 72)
    for size in SIZES:
        d_list, a_list, wins = [], [], 0
        for _ in range(MAPS_PER_SIZE):
            start, goal, walls = make_map(size, rng)
            d, _, _ = run_dstar(size, start, goal, walls)
            a, _, _ = run_fair_astar(size, start, goal, walls)
            d_list.append(d)
            a_list.append(a)
            wins += d < a
        dm, am = statistics.mean(d_list), statistics.mean(a_list)
        print(f"{size:>3}x{size:<2} | {dm:>16.0f} | {am:>16.0f} | "
              f"{(1 - dm / am) * 100:>8.0f}% | {wins:>5}/{MAPS_PER_SIZE}")


if __name__ == "__main__":
    main()