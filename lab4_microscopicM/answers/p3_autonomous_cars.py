# TMM Lab 04 - Problem 3: autonomous cars in a mixed fleet
#   (a) analytical capacity gain for three following rules
#   (b) Optimal Velocity Model simulation of the same rules
# Run: python p3_autonomous_cars.py  (~1 min)

# %% Imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

# %% (a) analytical
# At a fixed speed v, flow = v / (mean distance between cars). Each car keeps
# distance d (human rule) or d/2 (autonomous rule). With a fraction p of
# autonomous cars placed at random, the chance a given car keeps d/2 is
#   (i)   extreme:     it is autonomous                         -> p
#   (ii)  safe:        it AND the car behind are autonomous     -> p^2
#   (iii) super safe:  it, the car behind AND in front are      -> p^3
# Mean spacing = d (1 - q/2) with q = p, p^2, p^3, so
#   capacity ratio = 1 / (1 - q/2).
# A 25% increase needs 1/(1 - q/2) = 1.25  ->  q = 0.4:
#   (i) p = 0.4,  (ii) p = sqrt(0.4) = 0.632,  (iii) p = 0.4^(1/3) = 0.737
rules = {"extreme": 1, "safe": 2, "super safe": 3}
for name, k in rules.items():
    print(f"(a) {name:10s}: p for +25% capacity = {0.4 ** (1 / k):.3f}")


# Monte Carlo check: random sequences of cars
def short_gap(is_av, rule):
    """Which cars keep d/2. Car i follows car i+1 (index increases downstream)."""
    av = is_av
    behind = np.r_[False, av[:-1]]       # car i-1 is behind car i
    ahead = np.r_[av[1:], False]         # car i+1 is in front of car i
    if rule == "extreme":
        return av
    if rule == "safe":
        return av & behind
    return av & behind & ahead


rng = np.random.default_rng(0)
ps = np.linspace(0, 1, 21)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for name, k in rules.items():
    mc = [1 / np.mean(np.where(short_gap(rng.random(100_000) < p, name), 0.5, 1.0)) for p in ps]
    axs[0].plot(ps, 1 / (1 - ps**k / 2), label=f"{name}: 1/(1 - p^{k}/2)")
    axs[0].plot(ps, mc, "k.", ms=4)
axs[0].axhline(1.25, color="gray", ls=":")
axs[0].set_xlabel("fraction autonomous p"); axs[0].set_ylabel("capacity / all-human capacity")
axs[0].legend(); axs[0].set_title("(a) analytical (lines) and Monte Carlo (dots)")


# %% (b) Optimal Velocity Model simulation
def simulate(p, rule, inflow, entry="moving", T=400.0, warmup=100.0, dt=0.05, Lr=1000.0,
             detector=800.0, dc_human=50.0, tau=1.0, Lc=5.0, seed=0):
    """OVM (Bando et al.):  dv/dt = (V(d) - v)/tau,
       V(d) = v_des/2 [tanh(d - d_c) + tanh(d_c)],  d = distance to the car in front.
    v_des ~ U(30, 35) m/s as in the provided ovm_model.py. Autonomous cars use
    d_c/2 when the rule allows. Cars arrive at rate `inflow` and wait in an
    entry queue, so the car entering always knows the type of the car behind it.
      entry = "rest":   as in ovm_model.py - enter at x = 0 with v = 0 once the
                        previous car has moved Lc. Flow then measures how fast a
                        standing queue discharges (start-up from rest).
      entry = "moving": enter at speed v_des once the gap ahead is d_c + 3 m
                        (its equilibrium spacing). Flow then measures the road's
                        capacity at speed - what part (a) assumes.
    Returns the measured flow (cars/s) past `detector` after the warm-up."""
    rng = np.random.default_rng(seed)
    queue_types = list(rng.random(int(2 * inflow * T) + 50) < p)   # future arrivals (generous)
    arrivals, entered = 0, 0
    x = np.zeros(0); v = np.zeros(0); vd = np.zeros(0); av = np.zeros(0, bool)
    passed = 0
    for step in range(int(T / dt)):
        t = step * dt
        arrivals += rng.random() < inflow * dt                  # cars join the entry queue
        if arrivals > entered:
            new_av = queue_types[entered]
            behind_av = queue_types[entered + 1]                # next car in the queue
            ahead_av = av[0] if len(x) else False
            dc_new = dc_human / 2 if short_gap(np.array([behind_av, new_av, ahead_av]),
                                               rule)[1] else dc_human
            vdes = 30 + 5 * rng.random()
            room = (len(x) == 0 or x[0] > Lc) if entry == "rest" else                    (len(x) == 0 or x[0] > dc_new + 3)
            if room:
                x = np.r_[0.0, x]; v = np.r_[0.0 if entry == "rest" else vdes, v]
                vd = np.r_[vdes, vd]; av = np.r_[new_av, av]; entered += 1
        if len(x) == 0:
            continue
        gap = np.r_[np.diff(x), 1e6]                     # front-to-front distance
        # the most upstream car's follower is the next car in the queue
        behind = np.r_[queue_types[entered] if arrivals > entered else False, av[:-1]]
        ahead = np.r_[av[1:], False]
        if rule == "extreme":
            short = av
        elif rule == "safe":
            short = av & behind
        else:
            short = av & behind & ahead
        dc = np.where(short, dc_human / 2, dc_human)
        V = vd / 2 * (np.tanh(gap - dc) + np.tanh(dc))
        v = v + dt * (V - v) / tau
        x_new = x + dt * v
        x_new[:-1] = np.minimum(x_new[:-1], x_new[1:] - 0.01)   # no overtaking/crashes
        if t >= warmup:
            passed += np.sum((x < detector) & (x_new >= detector))
        x = x_new
        keep = x <= Lr
        x, v, vd, av = x[keep], v[keep], vd[keep], av[keep]
    return passed / (T - warmup)


ps_sim = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
styles = {"extreme": "o-", "safe": "s-", "super safe": "^-"}
# Provided entry rule (start from rest) for comparison:
q_rest = [simulate(p, "extreme", 5.0, entry="rest") for p in [0, 1]]
print(f"(b) start-from-rest entry, extreme rule: all-human {q_rest[0]:.3f}, all-AV "
      f"{q_rest[1]:.3f} cars/s -> ratio {q_rest[1] / q_rest[0]:.2f} (queue discharge, not capacity)")
for inflow, ls in [(5.0, "-"), (1.0, "--")]:
    base = simulate(0.0, "extreme", inflow)
    for name in rules:
        q = [simulate(p, name, inflow) for p in ps_sim]
        axs[1].plot(ps_sim, np.array(q) / base, styles[name][0] + ls,
                    label=f"{name}, inflow {inflow}/s")
        print(f"(b) inflow {inflow}/s, {name:10s}: flow {np.round(q, 3)} cars/s "
              f"(ratio at p = 1: {q[-1] / base:.2f})")
axs[1].axhline(1.25, color="gray", ls=":")
axs[1].set_xlabel("fraction autonomous p"); axs[1].set_ylabel("measured flow / all-human flow")
axs[1].legend(fontsize=7); axs[1].set_title("(b) OVM simulation, 1 km road")
fig.savefig(FIG / "p3_autonomous.png", dpi=130, bbox_inches="tight")

plt.show()
