"""Checks D* Lite against plain A* on random grids."""

import heapq
import random
from v2.dstar_lite import DStarLite, MOVES

SIZE = 10


def astar(size, start, goal, blocked):
    """Plain forward A*. Returns (path_length or None, expansions)."""
    def h(a):
        return abs(a[0] - goal[0]) + abs(a[1] - goal[1])

    open_heap = [(h(start), 0, start)]
    g = {start: 0}
    closed = set()
    expansions = 0
    while open_heap:
        _, cost, c = heapq.heappop(open_heap)
        if c in closed:
            continue
        closed.add(c)
        expansions += 1
        if c == goal:
            return cost, expansions
        for dx, dy in MOVES:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < size and 0 <= n[1] < size) or n in blocked:
                continue
            if cost + 1 < g.get(n, float("inf")):
                g[n] = cost + 1
                heapq.heappush(open_heap, (cost + 1 + h(n), cost + 1, n))
    return None, expansions


def random_map(rng, density=0.2):
    start, goal = (0, 0), (SIZE - 1, SIZE - 1)
    blocked = {(x, y) for x in range(SIZE) for y in range(SIZE)
               if rng.random() < density} - {start, goal}
    return start, goal, blocked


def test_static(n=20):
    rng = random.Random(1)
    passed = 0
    for i in range(n):
        start, goal, blocked = random_map(rng)
        a_len, _ = astar(SIZE, start, goal, blocked)
        d = DStarLite(SIZE, start, goal, blocked)
        d.compute_shortest_path()
        p = d.path()
        d_len = None if p is None else len(p) - 1
        ok = a_len == d_len
        passed += ok
        print(f"map {i:2d}: A* {a_len}  D* Lite {d_len}  {'OK' if ok else 'MISMATCH'}")
    print(f"\nStatic test: {passed}/{n} match\n")


def test_new_wall(n=20):
    """Robot walks its path; a wall appears on the path; both replan."""
    rng = random.Random(7)
    passed = 0
    for i in range(n):
        start, goal, blocked = random_map(rng, 0.15)
        d = DStarLite(SIZE, start, goal, blocked)
        d.compute_shortest_path()
        p = d.path()
        if p is None or len(p) < 6:
            continue
        # Robot takes 2 steps, then discovers a wall 2 cells ahead on its path.
        d.move_robot(p[2])
        new_wall = p[4]
        before = d.expansions
        d.add_walls([new_wall])
        d_cost = d.expansions - before
        q = d.path()
        d_len = None if q is None else len(q) - 1

        a_len, a_cost = astar(SIZE, p[2], goal, blocked | {new_wall})
        ok = a_len == d_len
        passed += ok
        print(f"map {i:2d}: A* len {a_len} ({a_cost} expanded)  "
              f"D* Lite len {d_len} ({d_cost} expanded)  {'OK' if ok else 'MISMATCH'}")
    print(f"\nNew-wall test: {passed} matched")


if __name__ == "__main__":
    test_static()
    test_new_wall()