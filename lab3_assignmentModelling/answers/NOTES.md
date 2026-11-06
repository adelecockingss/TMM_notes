# Lab 3 – Assignment Modelling: notes and answers

Sources: Eddie Wilson's assignment lectures (01, 02, C, D, E), `networkExampleLecture2021.m`, `AssignmentQ2.mlx`, `BigNetworkExperiments.mlx`. Lab sheet: `TMM202526Lab03.pdf`.

## 0. How to run

Run everything from this `answers/` folder.

| Questions | Python | MATLAB |
|---|---|---|
| Q1–Q3 two parallel links | `q1_q3_parallel.py` | `q1_q3_parallel.m` |
| Q4–Q5 n links, nonlinear costs | `q4_q5_parallel_n.py` | `q4_q5_parallel_n.m` |
| Q6–Q8 Braess | `q6_q8_braess.py` | `q6_q8_braess.m` |
| Q9 scissors network | `q9_scissors.py` | `q9_scissors.m` |
| Q10–Q14 big networks | `q10_q14_big_networks.py` (~45 s) | `q10_q13_big_networks.m` (~75 s) |

**Your MATLAB has no Optimization Toolbox, so there's no `quadprog`.** The MATLAB scripts call `quadprog` if it exists and otherwise use base-MATLAB solvers I wrote. The same solvers are in `assign_utils.py`:

| solver | use | idea |
|---|---|---|
| `parallel_affine` | parallel links, affine costs | exact "water-filling": find the common cost level C |
| `qp_small` | small QPs (≤ ~15 variables) | **the complementarity method, automated**: try each support, solve the KKT system |
| `gradient_projection` | one OD pair, any network | path-based; shifts flow from dear routes to the cheapest; exact |
| `frank_wolfe` (+ `aon_flows`) | any OD matrix, any network | link-based; each step is an all-or-nothing shortest-path load; slow to converge fully |

`data/bignet_edges.csv` and `bignet_nodes.csv` hold the graph from `BigNetworkExperimentsBetaSkeleton.mat`, exported with MATLAB because the `graph` object can't be read in Python.

---

## 1. Concepts

- **Assignment:** given fixed demand, how does traffic split between the routes (or modes)? Every option has a generalised **cost** (time, money, ...).
- **Link cost function** $c_i(x_i)$: the cost per user on link i when its flow is $x_i$. Throughout this lab it is **affine**: $c_i = a_i + b_ix_i$, where
  - a = free-flow cost;
  - b = "congestibility", roughly 1/capacity.

  More realistic is the BPR function $t_0(1 + 0.15(x/\text{cap})^4)$.
- **User Equilibrium (UE), Wardrop's first principle:** *all used routes have equal cost, and no unused route is cheaper.* Nobody can gain by switching alone (a Nash equilibrium).
- **System Optimum (SO):** minimise the total cost $f = \sum_i x_ic_i(x_i)$.
- **Price of Anarchy:** $\text{PoA} = f_{UE}/f_{SO} \ge 1$. For affine costs, **PoA ≤ 4/3** (Roughgarden–Tardos), attained by Pigou's network at d = 1.

### Two ways to compute UE

1. **Complementarity:** guess which links/routes are used, set their costs equal, solve, then check that the unused ones are no cheaper. This explodes combinatorially, with $2^n$ cases.
2. **Beckmann's trick:** UE is the minimiser of
   $$\hat f(x) = \sum_i \int_0^{x_i} c_i(s)\,ds$$
   subject to the same constraints. Its optimality (KKT) conditions *are* Wardrop's conditions. For affine costs, $\hat f = \sum (a_ix_i + \tfrac12 b_ix_i^2)$.

Both problems are quadratic programmes:

$$\min_x\; a^\top x + \theta\, x^\top \text{diag}(b)\, x,\quad \theta = \tfrac12\text{ (UE)},\; 1\text{ (SO)}$$

In `quadprog`'s form, $\tfrac12x^\top Hx + f^\top x$, this means $H = \text{diag}(b)$ for UE and $H = 2\,\text{diag}(b)$ for SO, with $f = a$.

**Useful equivalence:** SO = UE with each cost replaced by its **marginal cost** $\frac{d}{dx}[x\,c(x)] = a + 2bx$. Each user pays their own cost *plus* the extra delay they impose on everyone else. So a congestion toll of $b_ix_i$ per user turns UE into SO.

### Constraints

- **Route variables y:**
  - link flows: $x = Ay$, where A is the link–route incidence matrix;
  - demand: $My = d$, where M is the OD–route incidence matrix;
  - $y \ge 0$.
- **Link variables x:** node conservation $Bx = s$ (B = node–link incidence, s = sources/sinks) and $x \ge 0$. You never need to list the routes.

Link flows are **unique** at UE (because the problem is strictly convex in x). Route flows usually aren't, e.g. when there are more routes than links.

---

## 2. Two parallel links (Q1–Q3)

### Q1: $c_1 = \tfrac12 + x_1$, $c_2 = 1 + \tfrac12x_2$, $d = 1$

**(a) UE by complementarity.**

- *Link 1 only:* $c_1(1) = 1.5 > c_2(0) = 1$, so this case fails.
- *Link 2 only:* $c_2(1) = 1.5 > c_1(0) = 0.5$, which fails too.
- *Both links used:* $\tfrac12 + x_1 = 1 + \tfrac12(1 - x_1)$, so $\tfrac32x_1 = 1$, giving **$x_1 = 2/3$, $x_2 = 1/3$**. Both costs are 7/6, and $f_{UE} = 7/6$.

**(b) SO.**

$$f = x_1(\tfrac12 + x_1) + x_2(1 + \tfrac12x_2)$$

Substitute $x_2 = 1 - x_1$ and differentiate:

$$f'(x_1) = \tfrac12 + 2x_1 - 1 - (1 - x_1) = 3x_1 - \tfrac32 = 0 \;\Rightarrow\; x_1 = \tfrac12$$

This is interior, so it is the minimum (check the endpoints for safety: f(0) = 1.5, f(1) = 1.5). **$x_1 = x_2 = 1/2$, $f_{SO} = 9/8$.**

**(c)** **PoA = (7/6)/(9/8) = 28/27 ≈ 1.037.**

**(d) Beckmann.**

$$\hat f = \tfrac12x_1 + \tfrac12x_1^2 + x_2 + \tfrac14x_2^2$$

$$\hat f'(x_1) = \tfrac12 + x_1 - 1 - \tfrac12(1 - x_1) = \tfrac32x_1 - 1 = 0 \;\Rightarrow\; x_1 = 2/3$$

This agrees with (a).

**(e) General demand d.**

| d | UE | SO |
|---|---|---|
| $d \le \tfrac14$ | $x_1 = d$ | $x_1 = d$ |
| $\tfrac14 < d \le \tfrac12$ | $x_1 = d$ | $x_1 = \frac{1+2d}{6}$ |
| $d > \tfrac12$ | $x_1 = \frac{1+d}{3}$ | $x_1 = \frac{1+2d}{6}$ |

UE uses link 2 once $c_1(d) > c_2(0)$, i.e. $d > \tfrac12$. SO uses it once the *marginal* cost $\tfrac12 + 2d > 1$, i.e. $d > \tfrac14$.

PoA = 1 for d ≤ 1/4, rises to a **maximum of 12/11 ≈ 1.091 at d = 1/2**, then decays back towards 1 as d → ∞. At d = 1/2: $f_{UE} = \tfrac12$ and $f_{SO} = \tfrac13\cdot\tfrac56 + \tfrac16\cdot\tfrac{13}{12} = \tfrac{11}{24}$.

### Q2: $c_1 = a + x_1$, $c_2 = 1 + bx_2$

**(a) The problems:**

- UE: minimise $ax_1 + \tfrac12x_1^2 + x_2 + \tfrac b2 x_2^2$
- SO: minimise $ax_1 + x_1^2 + x_2 + bx_2^2$

Both subject to $x_1 + x_2 = d$ and $x_1, x_2 \ge 0$.

**Closed forms:**

- UE uses link 2 once $d > 1 - a$, and then $x_1 = \frac{1 - a + bd}{1+b}$.
- SO uses link 2 once $d > \frac{1-a}{2}$, and then $x_1 = \frac{1 - a + 2bd}{2(1+b)}$.

**Rescaling.** Any affine pair $\alpha_1 + \beta_1x$, $\alpha_2 + \beta_2x$ reduces to this form: measure cost in units of $\alpha_2$ and flow in units of $\alpha_2/\beta_1$. If $a > 1$, link 2 is cheaper when empty, so relabel the links to get $a \le 1$. PoA doesn't depend on the units.

**(c–d) Behaviour as d varies.**

- The two assignments coincide for $d \le (1 - a)/2$, where everything goes on link 1.
- They differ in between, where SO starts using link 2 earlier than UE does.
- PoA → 1 as d → ∞.

So scan d up to a few times $\max(1 - a, 1/b)$.

**(e–3) Maximum PoA.** Do a 1-D maximisation over d for each (a, b), then a grid over a, b ∈ [0, 1]. The maximum is **4/3 at a = b = 0**, i.e. Pigou's network at d = 1. This is the classical bound.

## 3. n links and nonlinear costs (Q4–Q5)

- **Water-filling** for n affine links: $x_i(C) = \max(0, (C - a_i)/b_i)$. Find C with $\sum_i x_i(C) = d$; the solution is exact on each linear piece once the links are sorted by $a_i$.
- Over 200 random 5-link networks, the worst PoA found was 1.32–1.28. It is never above 4/3.
- **Nonlinear costs.** For Pigou with $c_1 = x^p$, $c_2 = 1$:

  $$\text{PoA} = \left[1 - p(p+1)^{-(p+1)/p}\right]^{-1}$$

  This gives 4/3 for p = 1, **2.151 for p = 4 (BPR-like)**, and → ∞ as p grows. The numerical solver reproduces this to 4 decimal places.
- **Solving nonlinear cases:** use bisection on the common cost level (Python) or `fzero` on the equal-cost condition (MATLAB).

## 4. Braess (Q6–Q8)

Links: 1 O→N, 2 N→D, 3 O→S, 4 S→D, 5 N→S. Routes: N = {1, 2}, S = {3, 4}, M = {1, 5, 4}.

$$A = \begin{pmatrix}1&0&1\\1&0&0\\0&1&0\\0&1&1\\0&0&1\end{pmatrix}$$

### Q6(a, b) Lecture network: $c_1 = x_1$, $c_4 = x_4$, $c_2 = c_3 = 1$, $c_5 = 0$

By symmetry $y_N = y_S = y$, $y_M = m$ and $2y + m = d$, so

$$c_N = 1 + y + m, \qquad c_M = 2(y + m)$$

| demand | UE | cost per user |
|---|---|---|
| $d \le 1$ | all midtown ($c_M = 2d \le c_N = 1 + d$) | 2d |
| $1 \le d \le 2$ | $y = d - 1$, $m = 2 - d$ | 2 |
| $d \ge 2$ | midtown unused | 1 + d/2 |

Without the midtown link the cost per user is $1 + d/2$. **The paradox occurs for 2/3 < d < 2**: there $2d > 1 + d/2$ (when d ≤ 1) and $2 > 1 + d/2$ (when d < 2). The worst ratio is 4/3, at d = 1.

### Q6(c) Swapped costs: $c_1 = c_4 = 1$, $c_2 = c_3 = x$

The midtown route costs 1 + 0 + 1 = 2 regardless of flow. It is only used once N and S already cost 2, so the new link just adds capacity. **No paradox.**

### Q6(d) Midtown link with constant cost $a_5$

The same algebra gives a paradox for

$$\tfrac23(1 - a_5) < d < 2(1 - a_5),$$

so it can only happen if **$a_5 < 1$**.

### Q7 Sheffi: $c_1 = 10x_1$, $c_4 = 10x_4$, $c_2 = 50 + x_2$, $c_3 = 50 + x_3$, $c_5 = 10 + x_5$, d = 6

- **Without link 5:** 3 + 3 split, each route costs 30 + 53 = **83**; total 498.
- **With link 5, UE:** $y_N = y_S = y_M = 2$, giving $x_1 = x_4 = 4$, $x_2 = x_3 = x_5 = 2$. Then $c_N = 40 + 52 = 92 = c_M = 40 + 12 + 40$. Each user pays **92**; total 552.
- **SO with link 5** doesn't use it (total 498). At the 3/3 split, midtown's marginal cost is 60 + 10 + 60 = 130, more than the North route's 60 + 56 = 116.

Adding the link makes everyone worse off: **Braess's paradox**.

### Q8: $c_1 = c_4 = a + x$, $c_2 = c_3 = 1 + bx$, $c_5 = 0$

**Without midtown:**

$$C_0 = a + 1 + \frac{d(1 + b)}{2}$$

**With midtown** (symmetric UE):

- **Region I**, $a + d \le 1$: everyone takes midtown, $C_1 = 2(a + d)$. Paradox iff $d > \frac{2(1 - a)}{3 - b}$.
- **Region II**, $a + d > 1$ and $d(1 - b) < 2(1 - a)$: $y = \frac{a + d - 1}{1 + b}$ and $C_1 = 2(a + d - y)$. Paradox iff $C_1 > C_0$.
- **Region III**, otherwise: midtown is unused and $C_1 = C_0$.

These formulas match the numerical QP to 1e-15 (`q6_q8_braess.*`), and `figures/q6_q8_braess.png` maps the paradox region in (a, b) for d = 0.5, 1, 2. **The paradox needs the "cheap but congestible" links (1, 4) and the "expensive but uncongestible" links (2, 3) to be sufficiently different.**

## 5. Networks: route vs link variables (Q9)

The two-origin "scissors" network (`networkExampleLecture2021.m`), with OD demands $d_{14} = d_{24} = 1$:

| formulation | flows on links 1–5 |
|---|---|
| UE, route variables | 0.5, 0.5, 1, 0.5, 0.5 |
| UE, link variables | 0.5, 0.5, 1, 0.5, 0.5 |
| SO (either formulation; the "exercise" is H = 2 diag(b) with Bx = s) | 0.375, 0.375, 0.75, 0.625, 0.625 |

At UE, all four routes cost 4.5. The system costs are 9 (UE) and 8.875 (SO), so PoA = 1.014.

This works in link variables without "colouring" because the flows only **converge** on a single destination: nothing needs to be unmixed.

## 6. Big networks (Q10–Q14)

### Q10: the beta-skeleton network (100 nodes, 176 streets), node 1 → node 100

- ⚠️ **Bug/feature in the provided `.mlx`:** B gives each undirected street one orientation, and with LB = 0 traffic can only use it in that direction. **Fix:** represent each two-way street as two arcs (u → v and v → u). At d = 1 this lowers the UE system cost from 18.97 to 18.75.
- **(c) Low demand** (d = 0.001): all traffic takes the single shortest route (13 arcs), UE = SO and PoA = 1. As d grows, congestion spreads traffic over more routes: 21–23 routes at d = 1, about 50 at d = 3. SO spreads it slightly more (PoA 1.009 at d = 1).
- **(d) Link costs at UE.** Weight every link by $a_i + b_ix_i^{UE}$ and compute shortest paths from O. Every used arc then has zero "slack" ($\pi_u + c_e - \pi_v = 0$, to 2e-11), so **all used routes are shortest routes with equal cost** (26.582 at d = 3). That's Wardrop's principle, recovered from link variables alone.

### Algorithms

**Frank–Wolfe:**

```
x = AON(free-flow costs)
repeat
    c = a + b x                 (SO: a + 2 b x)
    y = AON(c)                  shortest-path load of all demand
    lam = argmin over [0, 1] of the objective at x + lam (y - x)
                                (closed form for quadratics)
    x = x + lam (y - x)
until relative gap (c.x - c.y) / c.x < tol
```

- It is simple and handles any OD matrix.
- But it converges slowly: after 2,000 iterations the link flows are still 6e-3 off.

**Gradient projection** (path-based) uses the exact affine step $\Delta = (c_p - c_q)/\sum_{e\in p\triangle q} b_e$ and converges to machine precision in seconds.

### Q11: grid, corner to corner

The free-flow answer is wildly non-unique (48,620 shortest paths). With congestion, the **UE link flows are unique**: symmetric about the diagonal, and heaviest at O and D.

### Q12: spreading O/D over an area

Add a dummy **zone (centroid) node** joined to all street nodes within 250 m by **connector** arcs (cost = length, no congestion). Traffic then enters at 8 points, and the maximum street flow drops from 0.50 to 0.22. This is how MSOA-level demand is loaded in practice.

### Q13: many OD pairs

In link variables, each **destination** needs its own flow vector $x^{(j)}$ with $Bx^{(j)} = s^{(j)}$ ("colouring", as on the scanned notes). The total flow is $x = \sum_j x^{(j)}$, and the constraint matrix is block-diagonal, `kron(eye(n_dest), B)`. For the 6×6 grid with all-to-all demand that's 4,320 variables and 1,296 constraints.

**Frank–Wolfe avoids all of this**, because its AON step assigns whole paths.

**Result:** PoA = 1 exactly here. Every arc has the same free-flow cost and only Manhattan-shortest routes get used, so all used routes have the same $\sum a$. Then "equal $\sum bx$" (UE) and "equal $\sum 2bx$" (SO) are the same condition. Heterogeneous free-flow costs are needed for PoA > 1.

### Q14: real map (Bristol junction graph from Lab 2)

- free-flow time = length / speed, with 30 or 50 km/h by road type;
- b chosen so that the cost doubles at 600 veh/h per lane;
- 3,000 veh/h from the SW corner to the NE corner.

AON uses 106 arcs; UE spreads the traffic over 638 arcs.

## 7. Stochastic UE (lecture E, not examined this year)

The **logit** choice model gives route shares

$$p_i = \frac{e^{-\theta c_i}}{\sum_j e^{-\theta c_j}}$$

so every route gets some traffic, and as θ → ∞ you recover Wardrop. There's no Beckmann trick: you have to solve the fixed point $y = d\cdot\text{softmax}(-\theta\,A^\top c(Ay))$ in route variables. For example, iterate with damping ($y \leftarrow (1 - \lambda)y + \lambda\,\text{new}$), or use `fsolve`.

## 8. Coursework-style patterns

- Write UE/SO as `H, f, Aeq, beq`, with `quadprog(H, f, [], [], Aeq, beq, zeros(n,1), [])`. UE: H = diag(b). SO: H = 2 diag(b). Both: f = a.
- `fval` from `quadprog` is the Beckmann value for UE, **not** the system cost. Compute the system cost as `x'*(a + b.*x)`.
- **Always verify a UE:** used route costs should be equal and unused ones no cheaper (or check zero slack on used arcs).
- Represent two-way streets as two arcs.
- Spread zone demand with connector arcs.
