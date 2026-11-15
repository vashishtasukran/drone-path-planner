import heapq

SIZE = 10  # grid is SIZE x SIZE x SIZE


def get_neighbors(blocked, pos):
    x, y, z = pos
    result = []
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < SIZE and 0 <= ny < SIZE and 0 <= nz < SIZE and (nx, ny, nz) not in blocked:
            result.append((nx, ny, nz))
    return result


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def astar(blocked, start, goal, stats=None):
    frontier = [(0, start)]
    came_from = {start: None}
    cost = {start: 0}
    explored = 0
    while frontier:
        explored += 1
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in get_neighbors(blocked, current):
            new_cost = cost[current] + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                heapq.heappush(frontier, (new_cost + heuristic(nxt, goal), nxt))
                came_from[nxt] = current
    if stats is not None:
        stats["explored"] = explored
    if goal not in came_from:
        return None
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


if __name__ == "__main__":
    blocked = {(5, y, z) for y in range(10) for z in range(10) if not (y == 0 and z == 0)}
    path = astar(blocked, (0, 5, 5), (9, 5, 5))
    print(len(path))