# Lab 4 – Microscopic Modelling: notes and answers

Lecture source: Anders Johansson's microsimulation slides (parts 1–5) and the papers in this folder: Helbing & Molnar 1995 (social force, *PhysRevE.51.4282*), Bando et al. 1995 (optimal velocity model, *PhysRevE.51.1035*), Seyfried et al. 2005 (fundamental diagram), and others. Lab sheet: `tmm_lab_04_R01.pdf`.

## 0. How to run

Python only (the lab is Python-based). Run from this `answers/` folder. The scripts import the provided `World` class from `../tmm_lab_04_code_shared/`.

| Problem | script | runtime |
|---|---|---|
| 1 corridor (FD, groups, bi-directional) | `p1_corridor.py` | ~2 min |
| 2 evacuation (door width, headways, two doors, ship, maze) | `p2_evacuation.py` | ~1.5 min |
| 3 autonomous cars (analytical + OVM) | `p3_autonomous_cars.py` | ~30 s |
| 4 shortest vs quickest path | `p4_shortest_vs_quickest.py` | ~2.5 min |

`ped_common.py` contains:

- a vectorised social force (with an optional external force);
- a `PotentialField` "flood fill" navigator;
- `run()`, which steps the world headless and records every exit time.

Set `SHOW = True` in a script to watch the animation (run it from a terminal, not a notebook).

---

## 1. Microscopic models: the ideas

**Agent-based modelling:** define the agents, define their interactions (agent–agent and agent–environment), iterate in time, and watch the macroscopic patterns *emerge* (lanes, clogging, stop-and-go waves).

Families of pedestrian models:

| | discrete space | continuous space |
|---|---|---|
| microscopic | cellular automata (floor-field CA) | **social force**, velocity-obstacle, heuristic/vision |
| macroscopic | queueing networks | fluid / continuum (Hughes) |

### Social force model (Helbing & Molnar)

$$\frac{d\mathbf v_\alpha}{dt} = \underbrace{\frac{v^0_\alpha\mathbf e_\alpha - \mathbf v_\alpha}{\tau}}_{\text{driving}} + \sum_{\beta\neq\alpha}\mathbf f_{\alpha\beta} + \sum_i\mathbf f_{\alpha i} + \boldsymbol\xi$$

- **Driving term:** relax towards the desired velocity (speed $v^0$, direction $\mathbf e$) over the time τ (≈ 0.5 s).
- **Pedestrian–pedestrian force:** $\mathbf f_{\alpha\beta} = [A_p\,\mathbb 1(\text{touching}) + A_s\,w(\varphi)\,e^{-d/B_s}]\,\mathbf n_{\alpha\beta}$.
  - The physical term acts on contact; the "social" term keeps people apart before they collide.
  - The anisotropy $w(\varphi) = \lambda + (1-\lambda)\frac{1+\cos\varphi}{2}$ means you react more to people in front of you.
- **Wall force:** the same form, using the nearest point on each wall segment.
- **Integration:** explicit Euler, $\mathbf v \mathrel{+}= \mathbf a\,dt$, $\mathbf x \mathrel{+}= \mathbf v\,dt$, with dt = 0.05 s (`World.update_positions`), and accelerations capped at 10 m/s².
- **Vectorising:** build all pairwise differences at once (`pos[:, None] - pos[None, :]`, shape N×N×2). This is ~100× faster than a double loop. Your `scenario_corridor.py` already does this.

### Navigation

- **Simple:** point at the destination's centroid. This fails as soon as there's a wall in the way.
- **Potential field / floor field:** spread a "wave" from the destination over a grid, with blocked cells near walls. This gives the walking distance to the exit from every cell; the desired direction is downhill.
  - I compute it with Dijkstra on the 8-connected grid (`ped_common.PotentialField`). Fast marching gives smoother (Euclidean) distances.
- **Quickest path:** make crowded cells "longer" (a dynamic field), or choose between discrete route options using estimated walking time plus waiting time (Problem 4).

### Validation (lecture part 3–4)

- **Macroscopic:** the fundamental diagram $J = \rho v$.
- **Microscopic:** fit to trajectory data (replace one person by the model and measure the deviation).
- **Phenomenological:** does the model reproduce lane formation, stripe formation, clogging at bottlenecks, and "faster is slower"?

---

## 2. Problem 1: corridor

### (a)(i) Fundamental diagram

Set the density by N in a periodic corridor (L = 10 m, w = 3 m), measure the mean speed after a 10 s warm-up, and compute the specific flow $J = \rho v$.

| ρ (ped/m²) | 0.1 | 0.5 | 1.0 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| provided model, v (m/s) | 1.00 | 1.00 | 0.93 | 0.63 | 0.35 |
| narrower v₀ spread, v | 1.36 | 1.10 | 0.92 | 0.58 | 0.33 |
| Weidmann (empirical), v | 1.34 | 1.30 | 1.06 | 0.61 | 0.33 |

The flow peaks at about 1.2–1.3 ped/(m·s) near 2 ped/m², as expected.

**At high density the provided model is good.** At low density it's too slow, for two reasons:

1. **Desired speeds are drawn from N(1.34, 0.37²)**, which is very wide. In a narrow corridor fast walkers are stuck behind slow ones (platooning), so the mean is set by the slow walkers. Using N(1.34, 0.1²) fixes most of this.
2. **The 2 m range of the social force** means people ahead push back even at moderate density.

**Things I tried that didn't work:** shortening the range to B = 0.3 m keeps people far too fast in crowds (flow ≈ 3.3 at 3 ped/m²), and changing the anisotropy or τ made the overall fit worse. Calibration is a trade-off. Report the quantity you care about.

### (b) Group of 4 (L = 50, w = 4, N = 20)

Add an attraction $A_g\,\hat{\mathbf d}$ towards each group member further away than 1.2 m, using the **periodic** distance $dx - L\,\text{round}(dx/L)$.

| attraction $A_g$ | group spread after 60 s |
|---|---|
| none | 15 m |
| 1 m/s² | 1.5 m |
| 3 m/s² | 1.3 m |

With the attraction, the group walks together at roughly its average desired speed.

### (c) Bi-directional corridor (L = 10, w = 2, N = 40)

The mean speed is symmetric about α = 0.5, where it's lowest: 0.14 m/s, against 0.69 m/s for one-way flow. Even 10% counter-flow costs 25% of the speed. Lanes can form (self-organisation), but at 2 ped/m² in a 2 m corridor head-on jams dominate.

## 3. Problem 2: evacuation

### ⚠️ Calibration matters

With the provided parameters, flow through a door is only about 0.3–0.6 ped/(m·s), far below the measured 1.3–1.9 (Seyfried et al.). Two causes:

- body **radius 0.4–0.5 m** (0.8–1 m wide; real shoulders are about 0.45–0.5 m);
- a 2 m-range social force: the whole queue pushes back.

Using radius 0.23–0.28 m and $A_s = 2$, $B_s = 0.3$ m gives **about 1.5 ped/(m·s)**. All Problem 2 results use this evacuation calibration.

**Exit choice:** each person aims at the nearest point of the door opening (0.35 m in from the frame), then walks straight out.

### (a)(i)

50 people through a 2 m door: about **20 s** (21.5 s for seed 0).

### (a)(ii) Width vs time

| w (m) | 0.8 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| T (s) | 44 | 33 | 23 | 20 | 17 |

Bottleneck theory: $T \approx T_{walk} + N/(J_sw)$, so T is linear in 1/w (see the fitted line in `p2a_door_width.png`). For wide doors the door stops being the bottleneck: arrivals limit the flow, so $J_s$ drops.

### (a)(iii) Time headways (exit time of person i minus person i−1)

| door width | mean headway μ (s) | std σ (s) | CV = σ/μ |
|---|---|---|---|
| 2.0 m | 0.31 | 0.32 | 1.05 |
| 1.5 m | 0.38 | 0.37 | 0.98 |
| 1.0 m | 0.58 | 0.51 | **0.87** |

**Why the CV falls for narrow doors:**

- **Narrow door:** a queue forms and people pass in single file at an almost constant rate, like a server with a fixed service time.
- **Wide door:** two or three people pass side by side (headways ≈ 0) mixed with gaps, and arrivals are partly random (Poisson-like, CV ≈ 1).

Under panic, real narrow exits clog in arches and the CV goes *up* ("faster is slower"). This calm calibration doesn't show that.

### (b) Two doors, closest exit

The source is in the top-left corner, with the doors on the right and the bottom.

| door width | one door | two doors (closest) |
|---|---|---|
| 1.0 m | 33 s | 23 s |
| 1.5 m | 23 s | 19 s |
| 2.0 m | 21 s | 18 s |

The gain is largest for narrow doors, where queueing dominates. "Closest door" is a crude rule: if the crowd splits unevenly, one door queues while the other sits idle. Quickest-exit choice (Problem 4) does better.

### (c) Rolling ship

Add an external acceleration $a_y = -A\sin(2\pi t/10)$, where $A \approx g\sin\theta_{max}$ reduced by how much people compensate.

| A (m/s²) | 0 | 1.0 (~6°) | 2.0 (~12°) |
|---|---|---|---|
| evacuation time (s) | 19.8 | 19.2 | 21.4 |

The effect is within the noise up to ~6°, and about +8% at ~12°.

**Why:** the crowd is pushed alternately towards the top and bottom walls, so the queue sways and people approach the door at an angle. Only about 2 roll cycles happen during the evacuation. Steeper tilts, or people who can't compensate, would be worse.

### (d) Maze

Use the potential field (0.1 m grid, 0.35 m wall clearance). The path from the start is 35.7 m. All 30 people get out, the last at 75 s. `p2d_maze.png` shows the field and the trajectories hugging the inside corners, as shortest paths do.

## 4. Problem 3: autonomous cars

### (a) Analytical

At a common speed, capacity ∝ 1/(mean spacing). The spacing is d, or d/2 when the rule allows, and $q$ = the probability that a car is allowed d/2:

| rule | q | p needed for +25% capacity |
|---|---|---|
| extreme (AV) | p | **0.40** |
| safe (AV and the car behind is AV) | p² | **0.632** |
| super safe (AV, the car behind and the car in front are AV) | p³ | **0.737** |

The capacity ratio is $1/(1 - q/2)$, so +25% needs q = 0.4. A Monte Carlo check on random car sequences agrees.

### (b) Optimal Velocity Model (Bando et al.)

$$\dot v = \frac{V(\Delta x) - v}{\tau}, \qquad V(\Delta x) = \frac{v_{des}}2\left[\tanh(\Delta x - d_c) + \tanh d_c\right]$$

with $d_c$ = 50 m for humans and 25 m for AVs when the rule allows.

**Two modelling issues I found and fixed:**

1. **Car types at the entrance.** The provided code creates a car only when it enters. The entering car then never has a car behind it, so the "safe" rules can never apply exactly where flow is limited. **Fix:** an entry queue, so the next car's type is known.
2. **Starting from rest.** Cars enter at v = 0, so the flow measures how fast a standing queue discharges (all-AV only 1.57× all-human). For road **capacity**, let cars enter at speed once the gap is $d_c$ + 3 m.

**Results (entering at speed):**

| | p = 0 | p = 0.4 | p = 0.8 | p = 1 |
|---|---|---|---|---|
| inflow 5/s, extreme | 0.58 | 0.68 | 0.86 | 1.04 cars/s |
| inflow 5/s, safe | 0.58 | 0.61 | 0.74 | 1.04 |
| inflow 5/s, super safe | 0.58 | 0.59 | 0.70 | 1.04 |

The same ordering as part (a), but somewhat smaller gains: at p = 1 the ratio is 1.78, not 2, because the spacing is $d_c$ + 3 m, which isn't exactly halved.

**At inflow 1/s** the all-human road is saturated (capacity 0.58 < 1), but AV-rich roads aren't. The flow then **caps at the demand**, ≈ 1 car/s (extreme rule at p = 1: 0.98, ratio 1.69). Beyond that point the extra capacity is unused. The safe rules also suffer at low demand: when the entry queue empties, the entering car has no follower and must keep the long gap. **So the relative benefit depends on the demand: capacity gains only show when the road is saturated.**

## 5. Problem 4: shortest vs quickest path

- Via the 0.9 m gap the route is 19.3 m; via the 2 m gap it is 40.4 m. Arrivals are 1 per second, measured over 90 s.
- **Quickest-path rule:** each person below the barrier re-chooses every 1 s:
  $$T_k = \frac{\text{distance via gap }k}{v_0} + \frac{\text{queue at gap }k}{J_s\,w_k}$$

| calibration | shortest path | quickest path |
|---|---|---|
| provided (bodies 0.8–1 m wide) | 19 arrived | **26 arrived (+38%)** |
| evacuation calibration (realistic bodies) | 82 | 82 |

With the provided parameters, the 0.9 m gap fits one person at a time and clogs. Sending people the long way round once the queue grows gets more of them through.

With realistic bodies, the gap passes about 1.3–1.5 people/s, which is above the arrival rate, so no queue forms and nobody switches.

**Lesson:** quickest-path routing pays off only when the short route's capacity is below the demand. The answer depends on calibration, so state your parameters.

## 6. Coursework-style patterns

- Vectorise the pairwise forces; use `use_graphics=False` for experiments and render only for checking.
- Measure after a warm-up, and average over several random seeds (reseed after `World()`, which fixes seed 42).
- Record exit times yourself (compare the set of pedestrian ids before and after each step).
- Use potential fields for any non-convex geometry; use separate fields per route option for route choice.
- Always compare against empirical data (Weidmann's FD, door flows of 1.3–1.9 ped/(m·s)) and report the calibration you used.

## 7. Comments on your own work

The `scenario_corridor.py` in `tmm_lab_04_code_shared/` has already been extended to answer Problem 1 (a)–(c) with a vectorised social force. It's the same approach as my `p1_corridor.py`, and its group-force and periodic-distance code is correct. Things to add:

- a comparison of your fundamental diagram with Weidmann's curve;
- the discussion of why low-density speeds are low (the wide desired-speed spread).

`lab_work.py` is empty. `ovm_model.py` removes cars from a list while iterating over it, which can skip cars or raise an index error; my `p3` simulation avoids this by filtering arrays.
