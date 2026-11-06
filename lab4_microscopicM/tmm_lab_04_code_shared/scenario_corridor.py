# Author: Anders Johansson (a.johansson@bristol.ac.uk)
# Description: This file contains functionality to simulate pedestrian crowds.
# Date: 2024-10-10

from world import World, normalise_vector, vector_length, get_nearest_position
import math
from functools import partial
import numpy as np
from matplotlib import pyplot as plt

# Stop the figure window jumping to the front every time world.render() updates it
plt.rcParams["figure.raise_window"] = False

# Redraw the animation every RENDER_EVERY time steps (dt = 0.05 s, so 10 = twice per simulated second).
# Drawing is the slowest part of the code: a bigger number runs faster but the animation looks jumpier.
RENDER_EVERY = 10

def pedestrian_initialisation(pedestrian, polygons, statistics):
    pass

def fit_window_to_corridor(world, corridor_width):
    """Crop the empty space above the corridor and make the window wide, so the pedestrians look bigger."""
    world.maxY = corridor_width + 2
    width = world.maxX - world.minX
    height = world.maxY - world.minY
    world.fig.set_size_inches(12, max(2.5, 12*height/width + 1))

def update_directions(pedestrians, boundaries, polygons):
    for i, ped in pedestrians.items():
        destination_polygon_centroid = polygons[ped.destination]["nodes"].mean(axis=0)
        ped.desired_direction = normalise_vector(destination_polygon_centroid - ped.pos)

def update_directions_bidirectional(pedestrians, boundaries, polygons):
    # Walk straight along the corridor (+x or -x) towards the destination side (used in 1b and 1c)
    for i, ped in pedestrians.items():
        destination_polygon_centroid = polygons[ped.destination]["nodes"].mean(axis=0)
        dx = destination_polygon_centroid[0] - ped.pos[0]
        ped.desired_direction = np.array([np.sign(dx), 0.0])

def process_interactions(pedestrians, boundaries, polygons):
    """Social force model. Same forces as the original double for-loop, but calculated for all
    pairs of pedestrians at once with numpy arrays (about 100x faster for 60-90 pedestrians)."""
    # Parameters
    tau = 0.5
    A_boundary_social, B_boundary_social = .5, 2
    A_boundary_physical = 50
    A_social, B_social = .5, 2
    A_physical = 4
    Lambda = 0.2

    peds = list(pedestrians.values())
    if len(peds) == 0:
        return
    pos = np.array([ped.pos for ped in peds])                       # (N, 2)
    vel = np.array([ped.vel for ped in peds])                       # (N, 2)
    radius = np.array([ped.radius for ped in peds])                 # (N,)
    v0 = np.array([ped.desired_walking_speed for ped in peds])      # (N,)
    e0 = np.array([ped.desired_direction for ped in peds])          # (N, 2)

    # Driving force towards the desired velocity
    acc = (v0[:, None]*e0 - vel) * (1/tau)

    # Pairwise forces from other pedestrians: element [i, j] is about ped i and ped j
    diff = pos[:, None, :] - pos[None, :, :]                        # pos_i - pos_j, (N, N, 2)
    distance = np.linalg.norm(diff, axis=2)                         # (N, N)
    tangent = diff / np.where(distance < 1e-6, 1.0, distance)[:, :, None]   # unit vector j -> i
    # Angular dependency (pedestrians react more to people in front of them)
    speed = np.linalg.norm(vel, axis=1)
    vel_dir = vel / np.where(speed < 1e-6, 1.0, speed)[:, None]
    cos_phi = np.einsum("ik,ijk->ij", vel_dir, -tangent)
    angular_dependency = Lambda + (1.0 - Lambda)*((1 + cos_phi)/2.0)
    # Physical force (when touching) + social force
    touching = distance <= radius[:, None] + radius[None, :]
    magnitude = A_physical*touching + A_social*angular_dependency*np.exp(-distance/B_social)
    np.fill_diagonal(magnitude, 0.0)                                # no force from yourself
    acc += (magnitude[:, :, None]*tangent).sum(axis=1)

    # Forces from boundaries
    for boundary in boundaries:
        start = np.array(boundary[:2], dtype=float)
        segment = np.array(boundary[2:], dtype=float) - start
        t = np.clip((pos - start) @ segment / (segment @ segment), 0, 1)
        d = pos - (start + t[:, None]*segment)                      # from nearest wall point to ped
        dist = np.linalg.norm(d, axis=1)
        tang = d / np.where(dist < 1e-6, 1.0, dist)[:, None]
        mag = A_boundary_physical*(dist <= radius) + A_boundary_social*np.exp(-dist/B_boundary_social)
        acc += mag[:, None]*tang

    for k, ped in enumerate(peds):
        ped.acc = acc[k].copy()


# question 1 b) - corridor with a group of 4 pedestrians that stay together
def periodic_separation(pos_i, pos_j, corridor_length):
    """Vector from ped i to ped j, taking the shortest way round the periodic x-boundary."""
    d = pos_j - pos_i
    d[0] = d[0] - corridor_length*round(d[0]/corridor_length)
    return d

def process_interactions_with_groups(pedestrians, boundaries, polygons,
                                     corridor_length, group_name="group", A_group=1.0, d_group=1.2):
    """Normal social force interactions, plus an attractive force between members of the same group.
    A_group: strength of the attraction (m/s^2)
    d_group: group members only attract each other when further apart than this (m)"""
    process_interactions(pedestrians, boundaries, polygons)
    for i, ped_i in pedestrians.items():
        if ped_i.group_name != group_name:
            continue
        for j, ped_j in pedestrians.items():
            if i != j and ped_j.group_name == group_name:
                d = periodic_separation(ped_i.pos, ped_j.pos, corridor_length)
                if vector_length(d) > d_group:
                    ped_i.acc += A_group*normalise_vector(d)

def group_spread(world, corridor_length, group_name="group"):
    """Largest distance between any two group members (m)."""
    members = [ped.pos for ped in world.pedestrians.values() if ped.group_name == group_name]
    spread = 0.0
    for a in range(len(members)):
        for b in range(a+1, len(members)):
            spread = max(spread, vector_length(periodic_separation(members[a], members[b], corridor_length)))
    return spread

def run_group_corridor(N=20, N_group=4, corridor_length=50, corridor_width=4, dt=0.05,
                       sim_time=60, A_group=1.0, use_graphics=True):
    """Run the corridor with N pedestrians, N_group of which are a group that attract each other.
    Returns the times and the group spread at each time."""
    world_definition = {
        "space": {
            "corridor": {"type": "rectangle", "coordinates": [0, 0, corridor_length, corridor_width], "colour": "gray", "add_boundaries": False},
            "right": {"type": "rectangle", "coordinates": [corridor_length+1, 0, corridor_length+2, corridor_width], "colour": "green", "add_boundaries": False},
        },
        "pedestrians": {
            "individuals": {"source": "corridor", "destination": "right", "colour": "red", "birth_rate": 99, "max_count": N - N_group},
            "group": {"source": "corridor", "destination": "right", "colour": "blue", "birth_rate": 99, "max_count": N_group},
        },
        "boundaries": [[0, 0, corridor_length, 0], [0, corridor_width, corridor_length, corridor_width]],
        "periodic_boundaries": {"axis": "x", "pos1": 0, "pos2": corridor_length},
        "functions": {
            "update_directions": update_directions_bidirectional,  # walk straight along +x
            # partial() fixes the extra arguments, so World can call it with (pedestrians, boundaries, polygons)
            "process_interactions": partial(process_interactions_with_groups, corridor_length=corridor_length, A_group=A_group),
            "pedestrian_initialisation": pedestrian_initialisation
        }
    }

    world = World(world_definition, use_graphics=use_graphics)
    if use_graphics:
        fit_window_to_corridor(world, corridor_width)
        world.fig.suptitle(f"N = {N}  (red = individuals, blue = group of {N_group})")

    times, spreads = [], []
    for i in range(int(sim_time/dt)):
        world.update(dt)
        if i%RENDER_EVERY==0:
            world.render()
        times.append(world.time)
        spreads.append(group_spread(world, corridor_length))
    return times, spreads


# question 1 c) - bi-directional corridor
def run_bidirectional_corridor(N, alpha, corridor_length=10, corridor_width=2, dt=0.05,
                               warmup_time=10, measure_time=30, use_graphics=True):
    """Run a bi-directional corridor where a fraction alpha of the N pedestrians walk
    left-to-right, and return the average walking speed along their desired direction."""
    N_right = int(round(alpha*N))   # walking left-to-right
    N_left = N - N_right            # walking right-to-left
    world_definition = {
        "space": {
            "corridor": {"type": "rectangle", "coordinates": [0, 0, corridor_length, corridor_width], "colour": "gray", "add_boundaries": False},
            "right": {"type": "rectangle", "coordinates": [corridor_length+1, 0, corridor_length+2, corridor_width], "colour": "green", "add_boundaries": False},
            "left": {"type": "rectangle", "coordinates": [-2, 0, -1, corridor_width], "colour": "green", "add_boundaries": False},
        },
        "pedestrians": {
            "left_to_right": {"source": "corridor", "destination": "right", "colour": "red", "birth_rate": 99, "max_count": N_right},
            "right_to_left": {"source": "corridor", "destination": "left", "colour": "blue", "birth_rate": 99, "max_count": N_left},
        },
        "boundaries": [[0, 0, corridor_length, 0], [0, corridor_width, corridor_length, corridor_width]],
        "periodic_boundaries": {"axis": "x", "pos1": 0, "pos2": corridor_length},
        "functions": {
            "update_directions": update_directions_bidirectional,
            "process_interactions": process_interactions,
            "pedestrian_initialisation": pedestrian_initialisation
        }
    }

    world = World(world_definition, use_graphics=use_graphics)
    if use_graphics:
        fit_window_to_corridor(world, corridor_width)
        world.fig.suptitle(f"alpha = {alpha}  (red = left-to-right: {N_right}, blue = right-to-left: {N_left})")

    n_warmup = int(warmup_time/dt)
    n_measure = int(measure_time/dt)
    speeds = []
    for i in range(n_warmup + n_measure):
        world.update(dt)
        if i%RENDER_EVERY==0:
            world.render()
        if i >= n_warmup:
            # Speed of each pedestrian in their own desired direction (positive = making progress)
            v = [ped.vel.dot(ped.desired_direction) for ped in world.pedestrians.values()]
            speeds.append(np.mean(v))

    return np.mean(speeds)



#====================

print("Running scenario_corridor.py ...")
RUN_1A = True
RUN_1B = True
RUN_1C = True


if RUN_1A:
    print("running a)")
    # 1(a)(i): run for several densities
    corridor_width = 3
    corridor_length = 10
    world_definition = {
        "space": {
            "corridor": {"type": "rectangle", "coordinates": [0, 0, corridor_length, corridor_width], "colour": "gray", "add_boundaries": False},
            "right": {"type": "rectangle", "coordinates": [corridor_length+1, 0, corridor_length+2, corridor_width], "colour": "green", "add_boundaries": False},
        },
        "pedestrians": {
            "group1": {"source": "corridor", "destination": "right", "colour": "red", "birth_rate": 99, "max_count": 0}
        },
        "boundaries": [[0, 0, corridor_length, 0], [0, corridor_width, corridor_length, corridor_width]],
        "periodic_boundaries": {"axis": "x", "pos1": 0, "pos2": corridor_length},
        "functions": {
            "update_directions": update_directions,
            "process_interactions": process_interactions,
            "pedestrian_initialisation": pedestrian_initialisation
        }
    }

    n_steps = 800     # 40 s with dt = 0.05
    n_warmup = 200    # ignore the first 10 s while pedestrians spawn and speed up

    results = []
    for density in [0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3]:
        N_pedestrians = int(density*(corridor_length*corridor_width))
        if N_pedestrians == 0:
            print(f"density = {density}   N = 0 - skipped (corridor too small for this density)")
            continue
        world_definition["pedestrians"]["group1"]["max_count"] = N_pedestrians
        world = World(world_definition, use_graphics=True)
        fit_window_to_corridor(world, corridor_width)
        world.fig.suptitle(f"N = {N_pedestrians}")

        avg_speeds = []
        for i in range(n_steps):
            world.update(0.05)
            if i%RENDER_EVERY==0:
                world.render()
            # Collect data
            if len(world.pedestrians)>0 and i>=n_warmup:
                vec = np.array([1.0, 0.0])   # walking direction (along the corridor)
                speeds = []
                for _, ped in world.pedestrians.items():
                    speed = vec.dot(ped.vel)
                    speeds.append(speed)
                avg_speeds.append(sum(speeds) / len(speeds))

        avg_speed = sum(avg_speeds) / len(avg_speeds)
        flow = density*avg_speed   # specific flow, ped/(m s)
        results.append((N_pedestrians, density, avg_speed, flow))
        print(f"density = {density}   N = {N_pedestrians}   avg speed = {avg_speed:.2f} m/s   flow = {flow:.2f} ped/m/s")

    # Plot density against number of pedestrians
    N_plot = [r[0] for r in results]
    density_plot = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_plot, density_plot, "-o", color="#2a6fdb", linewidth=2, markersize=8)
    # Label each point with its number of pedestrians
    for N_i, density_i in zip(N_plot, density_plot):
        ax.annotate(f"N = {N_i}", (N_i, density_i), textcoords="offset points",
                    xytext=(-8, 8), ha="right", fontsize=9)
    ax.set_xlabel("Number of pedestrians, N")
    ax.set_ylabel(r"Density, $\rho$ (ped/m$^2$)")
    ax.set_title(f"Corridor density vs N  (L = {corridor_length} m, w = {corridor_width} m)")
    ax.grid(True, color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    # Plot flow against density (fundamental diagram)
    flow_plot = [r[3] for r in results]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(density_plot, flow_plot, "-o", color="#2a6fdb", linewidth=2, markersize=8)
    # Label each point with its number of pedestrians (left of the line before the peak, right after it)
    i_peak = flow_plot.index(max(flow_plot))
    for k, (N_i, density_i, flow_i) in enumerate(zip(N_plot, density_plot, flow_plot)):
        if k < i_peak:
            ax2.annotate(f"N = {N_i}", (density_i, flow_i), textcoords="offset points",
                         xytext=(-8, 4), ha="right", fontsize=9)
        elif k == i_peak:
            ax2.annotate(f"N = {N_i}", (density_i, flow_i), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)
        else:
            ax2.annotate(f"N = {N_i}", (density_i, flow_i), textcoords="offset points",
                         xytext=(8, 4), ha="left", fontsize=9)
    ax2.set_xlabel(r"Density, $\rho$ (ped/m$^2$)")
    ax2.set_ylabel("Specific flow, J (ped/m/s)")
    ax2.set_title(f"Flow vs density  (L = {corridor_length} m, w = {corridor_width} m)")
    ax2.set_xlim(0, max(density_plot)*1.15)   # room for the labels
    ax2.set_ylim(0, max(flow_plot)*1.15)
    ax2.grid(True, color="0.9")
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()

    plt.close(world.fig)   # close the last simulation window
    plt.pause(0.5)         # draw the results graphs now; they stay open while the next part runs


if RUN_1B:
    print("running b)")
    # 1(b): L = 50 m, w = 4 m, N = 20, with 4 pedestrians in a group
    times, spreads = run_group_corridor(N=20, N_group=4, corridor_length=50, corridor_width=4)
    print(f"Group spread: start = {spreads[0]:.1f} m, end = {spreads[-1]:.1f} m")

    # Plot how spread out the group is over time
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, spreads, color="#2a6fdb", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Group spread (m)")
    ax.set_title("Largest distance between group members")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.pause(0.5)   # draw the results graph now; it stays open while the next part runs
    

if RUN_1C:
    print("running c..")
    # 1(c)(i): average walking speed as a function of alpha
    corridor_width = 2
    corridor_length = 10
    N = 40
    alpha_values = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]

    mean_speeds = []
    for alpha in alpha_values:
        mean_speed = run_bidirectional_corridor(N, alpha, corridor_length, corridor_width)
        mean_speeds.append(mean_speed)
        print(f"alpha = {alpha:.2f}   average speed = {mean_speed:.2f} m/s")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alpha_values, mean_speeds, "-o", color="#2a6fdb", linewidth=2, markersize=8)
    ax.set_xlabel(r"Fraction walking left-to-right, $\alpha$")
    ax.set_ylabel("Average walking speed (m/s)")
    ax.set_title(f"Bi-directional corridor  (L = {corridor_length} m, w = {corridor_width} m, N = {N})")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.pause(0.5)   # draw the results graph now; it stays open while the next part runs


# Keep all the results graphs open at the end (close them to finish the program)
plt.ioff()
plt.show()
