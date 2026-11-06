"""
TMM Lab 04 - shared pedestrian-model helpers, built on the lab's World class.

    social_force(...)          vectorised social force model (+ optional external force)
    PotentialField             'flood fill' distance-to-destination on a grid
    target_direction(...)      desired direction towards a point or along a field
    run(world, T, dt)          headless run that records every exit time
    rect(...), wall_with_door  geometry helpers

All scripts import World from ../tmm_lab_04_code_shared (the provided code).
"""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

CODE = Path(__file__).resolve().parents[1] / "tmm_lab_04_code_shared"
sys.path.insert(0, str(CODE))
from world import World, normalise_vector   # noqa: E402  (the provided simulator)

# Default social force parameters (lecture / provided code values)
PARAMS = dict(tau=0.5, A_social=0.5, B_social=2.0, A_physical=4.0, Lambda=0.2,
              A_wall_social=0.5, B_wall_social=2.0, A_wall_physical=50.0)


def social_force(pedestrians, boundaries, params=None, external=None):
    """Social force model (Helbing & Molnar) for all pedestrians at once:

      dv_a/dt = (v0 e_a - v_a)/tau                                  driving force
              + sum_b [A_p 1(touching) + A_s w(phi) exp(-d_ab/B_s)] n_ab   other peds
              + sum_walls [A_wp 1(touching) + A_ws exp(-d/B_ws)] n          walls
              + external                                           (e.g. ship roll)

    n_ab = unit vector from b to a; w(phi) = Lambda + (1-Lambda)(1+cos phi)/2
    makes people react more to those in front (anisotropy)."""
    p = dict(PARAMS, **(params or {}))
    peds = list(pedestrians.values())
    if not peds:
        return
    pos = np.array([q.pos for q in peds]); vel = np.array([q.vel for q in peds])
    rad = np.array([q.radius for q in peds])
    v0 = np.array([q.desired_walking_speed for q in peds])
    e0 = np.array([q.desired_direction for q in peds])

    acc = (v0[:, None] * e0 - vel) / p["tau"]

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    n = diff / np.where(dist < 1e-6, 1, dist)[:, :, None]
    speed = np.linalg.norm(vel, axis=1)
    vdir = vel / np.where(speed < 1e-6, 1, speed)[:, None]
    cos_phi = np.einsum("ik,ijk->ij", vdir, -n)
    w = p["Lambda"] + (1 - p["Lambda"]) * (1 + cos_phi) / 2
    touch = dist <= rad[:, None] + rad[None, :]
    mag = p["A_physical"] * touch + p["A_social"] * w * np.exp(-dist / p["B_social"])
    np.fill_diagonal(mag, 0)
    acc += (mag[:, :, None] * n).sum(1)

    for b in boundaries:
        s = np.array(b[:2], float); seg = np.array(b[2:], float) - s
        t = np.clip((pos - s) @ seg / (seg @ seg), 0, 1)
        dvec = pos - (s + t[:, None] * seg)
        dw = np.linalg.norm(dvec, axis=1)
        nw = dvec / np.where(dw < 1e-6, 1, dw)[:, None]
        acc += (p["A_wall_physical"] * (dw <= rad)
                + p["A_wall_social"] * np.exp(-dw / p["B_wall_social"]))[:, None] * nw

    if external is not None:
        acc += np.asarray(external)
    for k, q in enumerate(peds):
        q.acc = acc[k]


# ---------------------------------------------------------------------------
# Navigation: potential field ('flood fill' from the destination)
# ---------------------------------------------------------------------------
class PotentialField:
    """Distance-to-destination on a grid, avoiding walls.

    1. Grid of cells (size h) over the domain; cells within `clearance` of a
       wall are blocked.
    2. Dijkstra over the 8-connected grid (edge length h or h*sqrt(2)),
       starting from every cell inside the destination area. This is the
       lecture's 'wave spreading out from the destination'.
    3. Desired direction = downhill: towards the lowest-valued neighbouring
       cell (bilinear-interpolated gradient would be smoother).
    extra_cost (optional, per cell) lets us make crowded cells 'longer',
    which turns shortest paths into quickest paths (Problem 4)."""

    def __init__(self, xlim, ylim, walls, dest_box, h=0.2, clearance=0.3, extra_cost=None,
                 blocked_boxes=()):
        self.x0, self.y0, self.h = xlim[0], ylim[0], h
        self.nx = int(round((xlim[1] - xlim[0]) / h)); self.ny = int(round((ylim[1] - ylim[0]) / h))
        xs = self.x0 + h * (np.arange(self.nx) + 0.5); ys = self.y0 + h * (np.arange(self.ny) + 0.5)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P = np.column_stack([X.ravel(), Y.ravel()])
        blocked = np.zeros(len(P), bool)
        for b in walls:
            s = np.array(b[:2], float); seg = np.array(b[2:], float) - s
            t = np.clip((P - s) @ seg / max(seg @ seg, 1e-12), 0, 1)
            blocked |= np.linalg.norm(P - (s + t[:, None] * seg), axis=1) < clearance
        for (bx0, by0, bx1, by1) in blocked_boxes:
            blocked |= (P[:, 0] > bx0) & (P[:, 0] < bx1) & (P[:, 1] > by0) & (P[:, 1] < by1)
        self.blocked = blocked.reshape(self.nx, self.ny)
        cost = np.ones(len(P)) if extra_cost is None else 1 + extra_cost.ravel()
        idx = np.arange(len(P)).reshape(self.nx, self.ny)
        rows, cols, data = [], [], []
        for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            a = idx[max(0, -di):self.nx - max(0, di), max(0, -dj):self.ny - max(0, dj)]
            b = idx[max(0, di):, max(0, dj):][:a.shape[0], :a.shape[1]] if dj >= 0 else \
                idx[di:, :self.ny + dj][:a.shape[0], :a.shape[1]]
            a, b = a.ravel(), b.ravel()
            ok = ~blocked[a] & ~blocked[b]
            L = h * np.hypot(di, dj) * 0.5 * (cost[a] + cost[b])
            rows += [a[ok], b[ok]]; cols += [b[ok], a[ok]]; data += [L[ok], L[ok]]
        G = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(P), len(P)))
        bx0, by0, bx1, by1 = dest_box
        src = np.flatnonzero((P[:, 0] >= bx0) & (P[:, 0] <= bx1) & (P[:, 1] >= by0) &
                             (P[:, 1] <= by1) & ~blocked)
        D = dijkstra(G, indices=src, min_only=True)
        self.D = D.reshape(self.nx, self.ny)

    def cell(self, pos):
        i = np.clip(((pos[..., 0] - self.x0) / self.h).astype(int), 0, self.nx - 1)
        j = np.clip(((pos[..., 1] - self.y0) / self.h).astype(int), 0, self.ny - 1)
        return i, j

    def value(self, pos):
        i, j = self.cell(np.asarray(pos)); return self.D[i, j]

    def direction(self, pos):
        """Unit vector downhill from the 3x3 neighbourhood of pos's cell."""
        i, j = self.cell(np.asarray(pos, float))
        best, bv = np.zeros(2), self.D[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if (di or dj) and 0 <= ii < self.nx and 0 <= jj < self.ny:
                    v = self.D[ii, jj] + 0.0
                    if v < bv:
                        bv, best = v, np.array([di, dj], float)
        if not best.any():               # inside a blocked cell or at the goal
            return None
        return best / np.linalg.norm(best)


# ---------------------------------------------------------------------------
# Running a World without graphics and recording exit times
# ---------------------------------------------------------------------------
def run(world, T, dt=0.05, render_every=0, on_step=None, stop_when_empty_after=None):
    """Advance the world for T seconds. Records the time each pedestrian
    leaves (reaches its destination). Returns sorted exit times (s).
    stop_when_empty_after: stop early once this many have left and nobody is left."""
    exits, alive_prev = [], set()
    for k in range(int(round(T / dt))):
        if on_step is not None:
            on_step(world)
        world.update(dt)
        alive = set(world.pedestrians)
        exits += [world.time] * len(alive_prev - alive)
        alive_prev = alive
        if render_every and k % render_every == 0:
            world.render()
        if stop_when_empty_after and len(exits) >= stop_when_empty_after and not alive:
            break
    return np.array(sorted(exits))


def wall_with_door(x0, y0, x1, y1, door_centre, door_width):
    """A straight wall from (x0,y0) to (x1,y1) with a gap of door_width centred
    at distance door_centre along it. Returns two wall segments."""
    p, q = np.array([x0, y0], float), np.array([x1, y1], float)
    L = np.linalg.norm(q - p); u = (q - p) / L
    a = p + u * (door_centre - door_width / 2); b = p + u * (door_centre + door_width / 2)
    return [[*p, *a], [*b, *q]]
