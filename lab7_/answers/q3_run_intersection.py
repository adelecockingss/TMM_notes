# TMM Lab 06 - Q3: running the provided intersection model headless
# intersection_model.py runs a full simulation WITH graphics as soon as it is
# imported, so it is not imported here. Its Vehicle / RoadSimulation classes,
# pedestrian functions and main loop are copied below with the model logic
# UNCHANGED (only the render() method and graphics calls are dropped), and
# wrapped in run(). World is imported from the provided folder.
# BUG? comments flag things in the original code to discuss - not fixed.
# Run: python q3_run_intersection.py   (smoke test, case (i) only)

# %% Imports
import sys, math, random, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tmm_lab_06_microsimulation_code"))
from world import World, normalise_vector, vector_length, get_nearest_position  # noqa: E402
# World(definition, use_graphics=False) is supported: no figure, render() does nothing.
# NB World.__init__ calls random.seed(42) itself, so run() reseeds AFTER creating it.


# %% Car model (optimal-velocity model) - copied from intersection_model.py
class Vehicle:
    def __init__(self, x, v, v_desired, time_introduced=0):
        self.x, self.v, self.v_desired = x, v, v_desired
        self.time_introduced = time_introduced


class RoadSimulation:
    def __init__(self):
        self.time = 0
        self.vehicles = []                   # ordered: index 0 = most upstream car
        self.trajectories = {"t": [], "x": [], "density": [], "flow": []}
        self.completed_journeys = []
        self.inflow = 3.0                    # cars per second (overwritten in run)
        self.dt = 0.1
        self.Lc = 5                          # car length (m)
        self.Lr = 200                        # road length (m)
        self.tau = 1                         # relaxation time (s)
        self.d_c = 15                        # characteristic distance (m)
        self.intersection = {"position": self.Lr / 2, "is_green": True}

    def update(self, dt):
        self.time += dt
        # New car only if the entry is free - otherwise that arrival is LOST
        # (not queued), so the realised car inflow can be below cars_inflow.
        if random.random() < self.inflow * dt:
            if len(self.vehicles) == 0 or self.vehicles[0].x > self.Lc:
                speed = 10 + 2 * random.random()
                self.vehicles.insert(0, Vehicle(x=0, v=0, v_desired=speed, time_introduced=self.time))
        n = len(self.vehicles)
        for i in range(n - 1, -1, -1):
            d_alpha = self.vehicles[i + 1].x - self.vehicles[i].x if n > 1 and i < n - 1 else 999999.0
            d_alpha_light = (self.intersection["position"] - self.vehicles[i].x
                             if (not self.intersection["is_green"]) and self.vehicles[i].x < self.intersection["position"]
                             else 999999.0)
            v_e = self.vehicles[i].v_desired / 2 * (math.tanh(min(d_alpha, d_alpha_light) - self.d_c) + math.tanh(self.d_c))
            acc = (v_e - self.vehicles[i].v) / self.tau
            self.vehicles[i].v += dt * acc
            self.vehicles[i].x += dt * self.vehicles[i].v
            if i < n - 1 and self.vehicles[i].x > self.vehicles[i + 1].x - 0.01:
                self.vehicles[i].x = self.vehicles[i + 1].x - 0.01
            # (trajectories grow every step for every car - memory only)
            self.trajectories["t"].append(self.time)
            self.trajectories["x"].append(self.vehicles[i].x)
            self.trajectories["density"].append(len(self.vehicles) / self.Lr)
            self.trajectories["flow"].append(len(self.vehicles) / self.Lr * self.vehicles[i].v)
        # BUG? pop(i) inside 'for i in range(n)': the list shrinks while we loop,
        # so the next car is skipped and, if a car is popped before the end,
        # self.vehicles[i] can raise IndexError. Works in practice because
        # normally only the last (front) car leaves in a given step.
        for i in range(n):
            if self.vehicles[i].x > self.Lr:
                self.completed_journeys.append(self.time - self.vehicles[i].time_introduced)
                self.vehicles.pop(i)


# %% Pedestrian model (social force) - copied from intersection_model.py
# Module globals, read inside process_interactions (as in the original)
nearest_car_distance = 999
nearest_car_speed = 999


def pedestrian_initialisation(pedestrian, polygons, statistics):
    pass


def update_directions(pedestrians, boundaries, polygons):
    for i, ped in pedestrians.items():
        destination_polygon_centroid = polygons[ped.destination]["nodes"].mean(axis=0)
        ped.desired_direction = normalise_vector(destination_polygon_centroid - ped.pos)


def process_interactions(pedestrians, boundaries, polygons):
    tau = 0.5
    A_boundary_social, B_boundary_social = .5, 2
    A_boundary_physical = 50
    A_social, B_social = .5, 2
    A_physical = 4
    Lambda = 0.2
    for i, ped_i in pedestrians.items():
        ped_i.acc = (ped_i.desired_walking_speed * ped_i.desired_direction - ped_i.vel) * (1 / tau)
        for j, ped_j in pedestrians.items():
            if i != j:
                distance = vector_length(ped_i.pos - ped_j.pos)
                tangent = normalise_vector(ped_i.pos - ped_j.pos)
                vec_ij = normalise_vector(ped_j.pos - ped_i.pos)
                cos_phi = normalise_vector(ped_i.vel).dot(vec_ij)
                angular_dependency = Lambda + (1.0 - Lambda) * ((1 + cos_phi) / 2.0)
                if distance <= ped_i.radius + ped_j.radius:
                    ped_i.acc += A_physical * tangent
                ped_i.acc += A_social * angular_dependency * math.exp(-(distance) / B_social) * tangent
        for boundary in boundaries:
            pos_b = get_nearest_position(ped_i.pos, boundary)
            distance = vector_length(ped_i.pos - pos_b)
            tangent = normalise_vector(ped_i.pos - pos_b)
            if distance <= ped_i.radius:
                ped_i.acc += A_boundary_physical * tangent
            ped_i.acc += A_boundary_social * math.exp(-(distance) / B_boundary_social) * tangent
        # 'Kerb' wall at y = -3 switched on when a car is < 5 s and < 50 m away.
        # (Soft force only, from both sides: a pedestrian already on the road
        # is pulled/pushed relative to the kerb line too.)
        time_to_impact = nearest_car_distance / (0.0001 + nearest_car_speed)
        if time_to_impact < 5 and nearest_car_distance < 50:
            boundary = [0, -3, 1000, -3]
            pos_b = get_nearest_position(ped_i.pos, boundary)
            distance = vector_length(ped_i.pos - pos_b)
            tangent = normalise_vector(ped_i.pos - pos_b)
            ped_i.acc += 10 * A_boundary_social * math.exp(-(distance) / B_boundary_social) * tangent


# %% Scenario + run() wrapper
road_length, corridor_width, corridor_length = 200, 10, 10


def make_world_definition(peds_inflow):
    x0, x1 = road_length / 2 - corridor_width / 2, road_length / 2 + corridor_width / 2
    return {
        "space": {
            "start": {"type": "rectangle", "coordinates": [x0, -3 - corridor_length, x1, -3], "colour": "blue", "add_boundaries": False},
            "end": {"type": "rectangle", "coordinates": [x0, 3, x1, 3 + 2], "colour": "red", "add_boundaries": False},
        },
        "pedestrians": {"group1": {"source": "start", "destination": "end", "colour": "red",
                                   "birth_rate": peds_inflow, "max_count": 9999}},
        "boundaries": [[-10, -3 - corridor_length, -10, 3 + corridor_length], [210, -3 - corridor_length, 210, 3 + corridor_length]],
        "functions": {"update_directions": update_directions, "process_interactions": process_interactions,
                      "pedestrian_initialisation": pedestrian_initialisation},
    }


def run(cars_inflow, peds_inflow, T=600, seed=0, dt=0.05):
    """Run the intersection model headless for T seconds.
    Returns (mean vehicle journey time, mean pedestrian journey time, raw lists).
    Only COMPLETED journeys count (cars/people still in the system at T are ignored)."""
    global nearest_car_distance, nearest_car_speed
    world = World(make_world_definition(peds_inflow), use_graphics=False)
    random.seed(seed)                     # World() reset the seed to 42
    road = RoadSimulation()
    road.inflow = cars_inflow
    road.intersection = {"position": 100 - corridor_width / 2, "is_green": True}
    for _ in range(int(round(T / dt))):
        # BUG? 'crossing' = -5 <= y <= 3, but the road is y in [-3, 3] and the
        # start area ends at y = -3. So pedestrians WAITING in the last 2 m of
        # the pavement already turn the light red (and the y = -3 wall then
        # holds them there). Also the stop line is at x = 95, the crossing spans x 95-105.
        peds_currently_crossing = any(-5 <= p.pos[1] <= 3 for p in world.pedestrians.values())
        nearest_car_distance, nearest_car_speed = 999999, 999999
        for car in road.vehicles:
            if car.x <= road.intersection["position"]:
                dist = road.intersection["position"] - car.x
                if dist < nearest_car_distance:
                    nearest_car_distance, nearest_car_speed = dist, car.v
        road.intersection["is_green"] = not peds_currently_crossing
        world.update(dt)
        road.update(dt)
    veh = road.completed_journeys
    ped = world.get_statistics(["completed_journeys", "start", "end"])
    mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
    return mean(veh), mean(ped), veh, ped


# %% Smoke test: case (i) only
if __name__ == "__main__":
    t0 = time.time()
    mv, mp, veh, ped = run(0.1, 0.1, T=600, seed=0)
    print(f"case (i) 0.1/0.1: {len(veh)} cars, mean journey {mv:.1f} s | "
          f"{len(ped)} peds, mean journey {mp:.1f} s   [{time.time() - t0:.0f} s wall]")

# %% Q3b TODO: the three cases (i) 0.1, (ii) 0.2, (iii) 0.3 per second
# TODO: loop over the three inflows (one seed or several? how many?), and
#       tabulate mean journey times per mode. Which mode is ~constant, and why?
#       What does that suggest for signalled vs non-signalled crossings?
#       (Think: who effectively has priority in this model? Compare free-flow
#       times: 200 m at ~11 m/s for cars vs ~8 m at ~1.3 m/s for pedestrians.)
