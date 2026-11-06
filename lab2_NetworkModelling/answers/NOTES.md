# Lab 2 – Network Modelling: notes and answers

Lab sheet: `TMM202526Lab02.pdf`. There are no lecture slides in this folder, so the theory below is the standard material the lab builds on.

## 0. How to run

Run everything from this `answers/` folder.

| Question | Python | MATLAB | Runtime |
|---|---|---|---|
| Q1 grid graphs | `q1_grid.py` | `q1_grid.m` | seconds |
| Q2 perturbed grids | `q2_perturbed_grid.py` | `q2_perturbed_grid.m` | ~10 s |
| Q3 beta-skeletons | `q3_q4_beta_skeletons.py` | `q3_beta_skeletons.m` | ~1 min |
| Q4 torus | (same file as Q3) | `q4_torus.m` | ~1 min |
| Q5–Q8 OSM import | `q5_q11_osm.py` | `q5_readgeotable.m` (needs Mapping Toolbox, untested) | ~45 s first time |
| Q9–Q11 OSM analysis | `q5_q11_osm.py` | `q9_q11_osm.m` (reads the CSVs Python writes) | ~1 min |

Helper files:

- Python: `net_utils.py` (grid, edge flows, flow plots, beta-skeleton, metrics).
- MATLAB: `grid_network.m`, `edge_flows.m`, `plot_flows.m`, `beta_skeleton.m`.

Everything runs on **base MATLAB**. I didn't use the File Exchange ProximityGraphs toolbox: `beta_skeleton.m` replaces it, and its β follows the standard definition, not the toolbox's [−1, 1] convention.

---

## 1. Graph basics (Q1)

A **graph** is a set of nodes (junctions) plus a set of edges (streets), each edge with a weight (length or travel time).

- **Undirected** (`graph`): every street is two-way.
- **Directed** (`digraph`): each edge is one-way, u → v.

Storage options:

- **adjacency/weight matrix** A, with A(i, j) = weight. For networks it is sparse.
- **edge list**: rows (u, v, w). MATLAB's `G.Edges` table is exactly this.

### The lab's grid code

```matlab
A = -L*delsq(numgrid('S', n+2));   % 5-point Laplacian of an n x n grid
G = graph(A, 'omitselfloops');
```

`delsq` returns the discrete Laplacian: 4 on the diagonal and −1 for each grid neighbour. Times −L, that gives off-diagonal weights of +L. The diagonal (−4L) is removed by `'omitselfloops'`. Nodes are numbered **column by column**, so node k sits at x = L·floor((k−1)/n), y = L·rem(k−1, n).

### MATLAB ↔ Python cheat sheet

| task | MATLAB | Python |
|---|---|---|
| build | `graph(s, t, w)`, `digraph(...)` | `nx.Graph()` + `add_weighted_edges_from`, or a `scipy.sparse` matrix |
| neighbours, degree | `neighbors(G, k)`, `degree(G)` | `G.neighbors(k)`, `G.degree()` |
| one shortest path | `[p, d, ep] = shortestpath(G, s, t)` | `nx.shortest_path(G, s, t, weight=...)` |
| all distances | `distances(G)` | `scipy.sparse.csgraph.dijkstra(W)` |
| shortest-path tree | `shortestpathtree(G, s, 'OutputForm', 'vector')` | `dijkstra(W, indices=s, return_predecessors=True)` |
| edge index | `findedge(G, u, v)` | keep your own id matrix (`net_utils.to_csgraph`) |
| remove | `rmedge`, `rmnode` | `G.remove_edge`, or drop rows from the edge list |
| components | `conncomp(G, 'Type', 'strong')` | `connected_components(W, connection='strong')` |
| MST | `minspantree(G)` | `minimum_spanning_tree(W)` |
| highlight a path | `highlight(h, path, 'EdgeColor', 'r')` | draw the path's edges on top |

⚠️ **Indexing:** MATLAB counts from 1 and Python from 0. "Node 1 to 100" is 0 → 99 in Python.

### Q1(a) Degrees

On a 10×10 grid there are 4 corners with degree 2, 32 boundary nodes with degree 3 and 64 interior nodes with degree 4, giving 180 = 2n(n−1) edges.

### Q1(b) Shortest path 1 → 100

The length is 1.8 km (18 blocks). It is **not unique**: there are $\binom{18}{9} = 48{,}620$ equally short paths (choose which 9 of the 18 steps go right). Dijkstra just returns one of them. `edgepath` gives row numbers in `G.Edges`.

### Q1(c) All 10,000 shortest paths

- **Naive approach:** a double loop calling `shortestpath` for every pair (0.6 s in MATLAB).
- **Better approach:** one Dijkstra *per origin*, which gives the **shortest-path tree** `pred(v)`. Then push the demand back up the tree from the furthest nodes:

  ```
  load = demand(s, :)
  for v in nodes sorted by distance, furthest first:
      flow(edge pred(v) -> v) += load(v)
      load(pred(v))           += load(v)
  ```

  That's n Dijkstras instead of n² path listings. This is `edge_flows.m` / `net_utils.edge_flows`, and it's how real assignment codes do **all-or-nothing (AON) assignment**.

- **Sanity check** (always worth doing): $\sum_e f_e \ell_e = \sum_{s,t} d(s,t)$ = 6600 km. Total vehicle-km is the same whichever way you count it.
- With uniform all-to-all demand, the flow count is the (unnormalised) **edge betweenness centrality**.
- **Ties make the flows arbitrary.** The loop and the tree method give different edge counts (but the same totals), because they break ties differently. On a perfect grid, Dijkstra tends to put everything on the same "staircase", which is why the plain grid flow map looks lopsided. Fixes:
  - perturb the weights (Q2);
  - average over many tiny random perturbations (done in `q1_grid.py`);
  - split each OD demand equally over all tied paths, as your `qu1.py` does with the predecessor DAG. That is the cleanest approach.

### Q1(d) Hotness plots

Colour *and* line width ∝ flow:

- MATLAB: `plot(G, 'EdgeCData', flow, 'LineWidth', ...)`
- Python: a `LineCollection` with `array=flow`

The centre streets are the busiest, because most routes pass through the middle.

### Q1(e) Low Traffic Neighbourhoods

Modal filters let residents reach every house but stop **through traffic**.

**Design rule:** each filtered cell may be entered from **one side only**. My first attempt (four 2×2 cells) still allowed corner-cutting and only halved through traffic. The final design (two cul-de-sac halves) brings through traffic in the LTN from 6,048 trips to 0.

The cost: +17.5% total travel distance, and the boundary roads carry more traffic. This trade-off is the core of the 2024/25 coursework question.

---

## 2. Perturbed grids (Q2)

- **(a) Random weights U(0, 2L):** ties disappear, so the routes become unique and the flows no longer depend on tie-breaking.
- **(b) Jittered nodes ±L/2:** recompute every edge weight as the new Euclidean length, i.e. a new A matrix: `sparse([s; t], [t; s], [len; len])`.
- **(c) Closing streets:**
  - A connected graph needs at least N − 1 edges (a spanning tree), so at most 180 − 99 = 81 streets can close.
  - Protect a **minimum spanning tree** and remove the others at random. The mean path length rises slowly at first, then sharply once the network becomes tree-like (`figures/q2cd_closures_oneway.png`).
- **One-way streets:** reachability now means **strong connectivity**: every node can reach every other node along the arrows. Use `conncomp(G, 'Type', 'strong')` and check there is exactly one component.
  - **Robbins' theorem:** every street can be made one-way (in some orientation) iff the graph is **bridgeless**, i.e. every edge lies on a cycle.
  - My greedy random orientation reached 155–159 of 180 one-way streets before it got stuck. A depth-first-search orientation reaches all 180.
- **(d) Plotting one-way vs two-way:** grey thick lines for two-way streets; `quiver` arrows for one-way.
- **(e) Gravity demand** $T_{ij} \propto P_iP_j/d_{ij}^2$: short trips dominate, so the load concentrates around populous nodes rather than through the centre.

---

## 3. Beta-skeletons (Q3)

Place the nodes (junctions) first, then join p and q by a street **iff no other node lies in the forbidden region** $R(p, q, \beta)$:

| β | region | graph |
|---|---|---|
| β < 1 | thin lens: intersection of two discs of radius $d/2\beta$ through p and q | dense, tends to the complete graph as β → 0 |
| β = 1 | disc with diameter pq | **Gabriel graph** |
| 1 < β, lune | *intersection* of two discs of radius βd/2 centred on line pq | lune β = 2 is the **Relative Neighbourhood Graph** |
| 1 < β, circle | *union* of two discs of radius βd/2 passing through p and q | stays connected longer |

Properties:

- The region grows with β, so you get **fewer edges** (nested graphs: bigger β gives a subgraph).
- MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay. So for β ≥ 1 you only need to test the Delaunay edges, which is fast.
- Lune skeletons can disconnect for β > 2.

### Trade-off (`figures/q3_tradeoff.png`)

Results for 144 random nodes, averaged over 4 layouts:

| β | length / MST length | mean detour | max flow |
|---|---|---|---|
| 0.8 | 8.2 | 1.03 | 1,113 |
| 1.0 | 2.63 | 1.13 | 2,734 |
| 1.4 | 1.93 | 1.21 | 3,669 |
| 2.0 | 1.44 | 1.32 | 3,992 |

Raising β gives a cheaper network, but longer routes and more concentrated flow. Jittered-grid nodes give lower detours and lower peak flows than random nodes at the same β: grids are efficient.

Metrics worth reporting:

- **cost:** total length, normalised by the MST length;
- **performance:** mean shortest path, or mean **route factor/detour** = network distance ÷ straight-line distance;
- **congestion:** max edge flow, mean edge flow, and the Gini coefficient of flows.

## 4. Torus (Q4, Espinosa, Connors & Wilson)

In a square, the centre is a bottleneck: everything crosses it. Wrapping the square into a **doubly periodic torus** removes the boundary, so every location is statistically equivalent.

**Implementation:**

1. Copy the N points into a 3×3 block of tiles.
2. Build the skeleton of all 9N points.
3. Keep the edges with at least one end in the central tile.
4. Map both ends back with `mod N`, and store the edge length.

Straight-line distances use the **minimum image**: `dx = dx - W*round(dx/W)`.

**Results:** the same length/detour trade-off as in the square, but flow is spread much more evenly. Max/mean flow is about 2.2–2.8 on the torus, against about 3–4 in the square (for example, mean ≈ 1,000 and max ≈ 3,500 at β = 1.5). It *decreases* with β on the torus.

---

## 5. OpenStreetMap (Q5–Q11)

### The OSM data model

- **node** = point (id, lat, lon)
- **way** = ordered list of node ids + tags (`highway=residential`, `name`, `oneway=yes|-1`, `maxspeed`...)
- **relation** = group of nodes and ways (routes, turn restrictions)

Many nodes only trace the curve of a road; **junctions** are the nodes shared by two or more ways.

### Q5: reading the data

MATLAB's `readgeotable` (Mapping Toolbox) returns *shapes* and loses the node–way links. In Python, `pyosmium` streams the file. With `apply_file(..., locations=True)`, every way also gets its node coordinates. One pass over the 71 MB `.osm.gz` takes about 20–40 s, so **cache the result** as CSV.

### Q6: sources

- Geofabrik regional `.osm.pbf` extracts
- BBBike city extracts
- the Overpass API for small live areas (`osmnx.graph_from_point`)

PBF is binary protobuf: roughly 10× smaller than XML and much faster to read.

### Q7: `osmium` command-line tool

```
osmium fileinfo -e f.pbf                                 # counts, extent
osmium extract -b W,S,E,N in.osm.gz -o box.pbf           # cut a bounding box
osmium tags-filter box.pbf w/highway -o roads.pbf        # keep only roads
osmium cat in.osm -o out.pbf                             # convert format
```

On Windows: `conda install -c conda-forge osmium-tool`, or use WSL. `pyosmium` does the same jobs from Python, as your `qu5_onwards.py` shows.

### Q8: tables

- **nodes:** id, lon, lat, x_km, y_km. Use a local flat projection: x = Δlon·111.32·cos(lat₀), y = Δlat·111.32.
- **ways:** id, highway, name, oneway, node_ids.

### Q9: junction graph

**Junctions** are nodes used by two or more ways, plus way end points. Walk along each way, cutting it at every junction; the edge length is the **arc length**, the sum of the small segments. Direction comes from the `oneway` tag:

- `yes` → u → v only
- `-1` → v → u only
- anything else → both directions

| | OSM nodes | junctions | street edges | total length |
|---|---|---|---|---|
| Bristol (3×3 km box) | 8,295 | 2,192 | 2,748 | 169 km |
| Bath (3×3 km box) | 5,527 | 1,238 | 1,481 | 115 km |

(Whole ways that cross the box edge are kept, so the totals include street just outside the box.)

The degree-2 "junctions" are where one way ends and the next continues (a name change, say). You could merge them for a cleaner graph; `osmnx.simplify_graph` does exactly that.

### Q10: closures and new links

On the SW → NE route across Bristol (4.91 km; straight line 3.41 km, so the detour factor is 1.44):

- 3 of the route's streets are **bridges**: closing one disconnects the trip.
- The worst finite closure (Norrisville Road) adds 262 m.
- A hypothetical new 1.06 km straight street cuts the route to 4.65 km.

### Q11: are beta-skeletons realistic?

Compute the skeleton of the real junction points, then compare edge sets:

- **precision** = share of skeleton edges that are real streets;
- **recall** = share of real streets that the skeleton contains;
- **F1** = 2PR/(P + R).

Results:

| city | best β (lune) | F1 |
|---|---|---|
| Bristol | 1.5 | 0.68 |
| Bath | 2.4 | 0.68 |

Bath's streets are sparser and more tree-like, which suits a higher β. So yes, different cities have different optimal β.

**Caveats:**

- Real streets are curved, cross over each other (bridges, tunnels) and follow topography (the Avon Gorge!). A planar skeleton can't capture that.
- Dead ends and box-edge effects reduce recall.

---

## 6. Where this leads

The coursework-style pipeline:

1. map snippet;
2. census demand spread over the MSOA's nodes;
3. AON shortest routes (`edge_flows`);
4. busy edges.

AON ignores **congestion**. If every car takes the free-flow shortest path, the loaded links get slow and the routes are no longer shortest. Fixing this means finding an **equilibrium** assignment, which is Lab 3.

## 7. Comments on your own Lab 2 scripts

Your scripts are good and go beyond what the sheet asks:

- tie-splitting via the predecessor DAG;
- brute-force verification of the Gabriel and RNG graphs;
- SD vs SE bands over layouts;
- a square vs torus comparison.

Details:

- **Numbering:** your `qu1.py` is labelled "Q3(a)", but it is the sheet's Q1(c–d). Q2 (perturbed grid, closures, one-way streets) is covered here in `q2_perturbed_grid.*`.
- **Jitter:** `jittered_grid` uses Gaussian jitter with σ = 0.2L, while the sheet asks for uniform ±L/2. Both are fine, but say which one you used.
- **Q9–Q11:** these were TODOs in `qu5_onwards.py` and are now done in `q5_q11_osm.py`. Your `data/bs8_nodes.csv` / `bs8_ways.csv` have the same format as my tables, so `junction_graph()` works on them directly.
- **Runtime:** `qu4.py` and `qu3_compare.py` rely on pure-Python Dijkstra with tie-splitting, which is slow for large N. `scipy.sparse.csgraph.dijkstra` + tree accumulation is about 10× faster when you don't need tie-splitting.
