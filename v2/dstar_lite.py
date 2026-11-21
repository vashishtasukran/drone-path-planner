"""
D* Lite on a 2D grid (basic version, Koenig & Likhachev 2002, Figure 3 with km).

The search runs BACKWARDS from the goal. Each cell stores:
  g   = its current best-known distance to the goal
  rhs = what g SHOULD be, based on its neighbours (one-step lookahead)
A cell is "consistent" when g == rhs. Only inconsistent cells sit in the queue.
When a wall appears, only cells whose rhs changes get fixed, so old work is reused.
"""

import heapq

INF = float("inf")
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 4-connected grid


class DStarLite:
    def __init__(self, size, start, goal, blocked=None):
        self.size = size
        self.start = start            # where the robot is now
        self.goal = goal
        self.last = start             # robot position at the last replan (for km)
        self.blocked = set(blocked or [])
        self.km = 0                   # key modifier: avoids re-sorting the queue when the robot moves
        self.g = {}
        self.rhs = {}
        self.heap = []                # entries: (k1, k2, cell)
        self.in_queue = {}            # cell -> its current valid key
        self.expansions = 0           # cells popped and processed

        self.rhs[goal] = 0
        self._push(goal)

    # ---------- grid helpers ----------
    def in_bounds(self, c):
        return 0 <= c[0] < self.size and 0 <= c[1] < self.size

    def neighbors(self, c):
        for dx, dy in MOVES:
            n = (c[0] + dx, c[1] + dy)
            if self.in_bounds(n):
                yield n

    def cost(self, a, b):
        # Moving into or out of a wall is impossible.
        if a in self.blocked or b in self.blocked:
            return INF
        return 1

    def h(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan

    def get_g(self, c):
        return self.g.get(c, INF)

    def get_rhs(self, c):
        return self.rhs.get(c, INF)

    # ---------- priority queue (heapq + lazy deletion) ----------
    def key(self, c):
        m = min(self.get_g(c), self.get_rhs(c))
        return (m + self.h(self.start, c) + self.km, m)

    def _push(self, c):
        k = self.key(c)
        self.in_queue[c] = k
        heapq.heappush(self.heap, (k[0], k[1], c))

    def _remove(self, c):
        # heapq can't delete, so forget the cell; its old heap entry becomes stale.
        self.in_queue.pop(c, None)

    def _clean_top(self):
        # Throw away stale entries until the top is a real, current one.
        while self.heap:
            k1, k2, c = self.heap[0]
            if self.in_queue.get(c) == (k1, k2):
                return
            heapq.heappop(self.heap)

    def top_key(self):
        self._clean_top()
        if not self.heap:
            return (INF, INF)
        k1, k2, _ = self.heap[0]
        return (k1, k2)

    def _pop(self):
        self._clean_top()
        k1, k2, c = heapq.heappop(self.heap)
        del self.in_queue[c]
        return (k1, k2), c

    # ---------- core algorithm ----------
    def update_vertex(self, u):
        # Recompute what g(u) SHOULD be from its neighbours.
        if u != self.goal:
            self.rhs[u] = min(
                (self.cost(u, s) + self.get_g(s) for s in self.neighbors(u)),
                default=INF,
            )
        self._remove(u)
        if self.get_g(u) != self.get_rhs(u):  # inconsistent -> needs work
            self._push(u)

    def compute_shortest_path(self):
        while (self.top_key() < self.key(self.start)
               or self.get_rhs(self.start) != self.get_g(self.start)):
            if not self.heap:
                break
            k_old, u = self._pop()
            self.expansions += 1
            k_new = self.key(u)

            if k_old < k_new:
                # Key was outdated (robot moved). Put it back with the fresh key.
                self.in_queue[u] = k_new
                heapq.heappush(self.heap, (k_new[0], k_new[1], u))
            elif self.get_g(u) > self.get_rhs(u):
                # Overconsistent: a shorter route was found. Lock it in.
                self.g[u] = self.rhs[u]
                for s in self.neighbors(u):
                    self.update_vertex(s)
            else:
                # Underconsistent: the old route got worse (e.g. a new wall).
                # Reset and let neighbours recompute.
                self.g[u] = INF
                self.update_vertex(u)
                for s in self.neighbors(u):
                    self.update_vertex(s)

    # ---------- using the result ----------
    def next_step(self, c):
        # Move to the neighbour with the lowest cost + g.
        best, best_val = None, INF
        for s in self.neighbors(c):
            v = self.cost(c, s) + self.get_g(s)
            if v < best_val:
                best, best_val = s, v
        return best

    def path(self):
        if self.get_g(self.start) == INF:
            return None
        p, c = [self.start], self.start
        while c != self.goal:
            c = self.next_step(c)
            if c is None or len(p) > self.size * self.size:
                return None
            p.append(c)
        return p

    def move_robot(self, new_start):
        self.start = new_start

    def add_walls(self, cells):
        """Call when the robot discovers new walls. Repairs only what changed."""
        self.km += self.h(self.last, self.start)
        self.last = self.start
        for c in cells:
            if c in self.blocked:
                continue
            self.blocked.add(c)
            self.update_vertex(c)
            for n in self.neighbors(c):
                self.update_vertex(n)
        self.compute_shortest_path()