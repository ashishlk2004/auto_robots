# Autonomous Robots — Hazardous-Barrel Collection

Coursework submission for an Autonomous Robots module, where I have implemented a
decentralised control solution for a team of up to three TurtleBot3 Waffle Pi robots that
search a walled arena for contaminated (red) and clean (blue) barrels, tow them to green
collection zones, and visit a cyan decontamination zone to shed the radiation they pick up
along the way — all in ROS 2 Humble and Gazebo Classic.

Each robot runs the same `robot_controller` node: a 10 Hz finite state machine layered on top
of the Nav2 stack, with a built-in waypoint navigator it falls back to when Nav2 cannot cope.
There is no central planner — robots coordinate by publishing target claims and barrel
sightings on two shared topics, so the team scales from one to three robots by launch
parameter alone.

---

## Highlights

**Hybrid deliberative / reactive control.** Nav2 handles long-range motion through the
`NavigateToPose` action; the FSM handles everything that needs tight sensor feedback —
visual servoing onto a barrel, a rear-LiDAR docking sequence, creeping over a zone boundary.

**Dual navigator with graceful degradation.** When Nav2 rejects goals or repeatedly fails to
plan — which it reliably does once a robot has reversed into a ten-barrel cluster and is
enclosed by lethal costmap cells — the controller switches to an internal waypoint-graph
navigator (Dijkstra over a fixed 25-node graph) with reactive LiDAR avoidance. The same
navigator is used end to end when the system is launched with `use_nav2:=false`.

**Fully decentralised coordination.** Two custom topics carry everything the team must agree
on: `/barrel_sightings` (shared world memory of detected barrels) and `/robot_status`
(state, pose, claimed target, destination zone). Robots avoid each other's claims and
each other's zones without a coordinator node.

**Offline-rasterised map.** `solution/config/map2.pgm` is generated from the world's collision
meshes by [`solution/tools/generate_map.py`](solution/tools/generate_map.py), so the map frame
coincides with the Gazebo world frame and AMCL poses can be compared directly against known
zone coordinates.

**Fault tolerance for the required failure classes.** Bounded re-align-and-retry cycles then
blacklisting on failed pick-ups; two independent stuck watchdogs (8 s without wheel motion,
plus 0.35 m of real displacement required every 25 s to catch rotation-in-place oscillation);
an `ESCAPE` recovery that turns toward the most open LiDAR direction and drives clear; and
re-grab-and-retry on a different lane when a drop misses the zone boundary.

**Contamination awareness.** Radiation accrues while towing a red barrel, so a contaminated
robot detours to the cyan zone after each delivery, and a tunable `red_penalty` biases target
selection toward clean barrels.

---

## Repository Layout

Only `solution/` and `solution_interfaces/` are my work. Everything else is the provided
assessment environment and is unmodified.

| Path | Origin | Contents |
| --- | --- | --- |
| `solution/` | **mine** | The controller, launch file, map, Nav2 parameters, offline tools |
| `solution_interfaces/` | **mine** | `RobotStatus.msg`, `BarrelSighting.msg` for coordination |
| `assessment/` | provided | Simulation environment, worlds, barrel manager, visual sensor |
| `assessment_interfaces/` | provided | Barrel / zone / item message definitions |
| `auro_interfaces/` | provided | `ItemRequest` service |
| `gazebo_ros_link_attacher/` | provided | Gazebo plugin used to attach a towed barrel |
| `.devcontainer/` | provided + mine | Dev container configs; the five scenarios under `customizations.auro` are mine |
| `rcutil.py` | provided | Workspace build / scenario / submission utility |
| `rosgraph.png` | mine | [ROS graph](rosgraph.png) of the running system (`rqt_graph` export) |

Inside the solution package:

```
solution/
├── launch/solution_launch.py                    # brings up the whole system
├── solution/robot_controller.py                 # the FSM controller (one per robot)
├── solution/data_logger.py                      # provided logger, unmodified
├── config/map2.pgm, map2.yaml                   # occupancy grid for AMCL + global costmap
├── config/initial_poses.yaml                    # optional override (see note below)
├── config/custom_rviz_windows.yaml              # optional override (see note below)
├── params/custom_nav2_params_namespaced.yaml    # Nav2 tuning, changes annotated inline
└── tools/generate_map.py, export_rosgraph.py    # offline, not needed at runtime
```

The two `config/` overrides are **not** used unless asked for: `solution_launch.py` defaults to
`initial_pose_package:=assessment`. Launch with `initial_pose_package:=solution` to use them.

The four small obstacles in the west room are deliberately absent from the map — they are
toggleable via the `obstacles` parameter, so the costmap obstacle layer has to find them
from LiDAR.

---

## Quick Start

### 1. Open the dev container

The workspace targets ROS 2 Humble on Ubuntu 22.04 with Gazebo Classic 11, supplied as a dev
container image (`ghcr.io/uoy-robostar/ros2-tb3/auro-dev:latest`). With Docker and the VS Code
[Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
installed, open the command palette, run **Dev Containers: Clone Repository in Container
Volume**, and point it at this repository.

Four configurations are available:

| Config | Use it for |
| --- | --- |
| `.devcontainer/devcontainer.json` (`auro-vnc`) | The default. A VNC desktop at <http://localhost:6080/> — works everywhere |
| `.devcontainer/wsl/devcontainer.json` | Windows + WSL2, GUI apps native on the desktop, GPU-accelerated via OpenGL→Direct3D |
| `.devcontainer/other/devcontainer.json` | Linux with native X11 (extra setup required — see [X11 on Linux](#x11-on-linux)) |
| `.devcontainer/vslam-*/devcontainer.json` | Provided VSLAM variants, unused by this solution |

On Apple Silicon, pull the amd64 image once before first use:

```bash
docker pull --platform linux/amd64 ghcr.io/uoy-robostar/ros2-tb3/auro-dev:latest
```

### 2. Build

From the workspace root:

```bash
./rcutil.py build
```

(equivalently `colcon build --symlink-install`, then `source install/setup.bash`)

### 3. Run a scenario

```bash
./rcutil.py list-scenarios
```

```bash
./rcutil.py run-scenario 1
```

Or launch directly with any combination of parameters:

```bash
ros2 launch solution solution_launch.py num_robots:=3 use_rviz:=false
```

Every parameter of the original assessment launch file is supported.

---

## Scenarios

Five configurations are defined under `customizations.auro.scenarios` in
`.devcontainer/devcontainer.json`. They vary team size, perception quality, odometry source,
item layout, obstacles, and the navigation stack itself.

| # | Name | Robots | Configuration | What it validates |
| --- | --- | --- | --- | --- |
| 1 | Baseline | 1 | Nav2, no noise | The full search→collect→deliver→decontaminate pipeline, no confounds |
| 2 | Team | 3 | Nav2 | Decentralised claiming, zone allocation, corridor avoidance at maximum team size |
| 3 | Noisy | 2 | `sensor_noise:=true`, encoder odometry | AMCL drift correction and rejection of noisy detections |
| 4 | Generality | 2 | `random_seed:=42`, `obstacles:=false`, `red_penalty:=2.0` | Not over-fitted to one barrel layout; contamination-avoidance bias |
| 5 | Degraded nav | 1 | `use_nav2:=false`, WORLD odometry | The fallback waypoint navigator; graceful degradation without Nav2 |

---

## Launch parameters

The solution adds two parameters to the provided set, both passed through to every
`robot_controller`:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `decon_threshold` | int | `1` | Contamination level at which a robot detours to the decontamination zone after a delivery. Contamination rises by 1 per second while towing a red barrel, so the default sends any robot that has towed one |
| `red_penalty` | float (m) | `0.0` | Extra path cost added to red barrels during target selection, biasing robots toward clean ones. `0.0` leaves selection purely nearest-first |

The solution performs best with the default `use_nav2:=true`; with `use_nav2:=false` every
controller uses its internal fallback navigator instead.

---

## Results

Fifteen missions — five scenarios × three repeats — each 840 s in a fresh container with a
fresh build, at real-time factor 0.85–1.00.

| Metric | Value |
| --- | --- |
| Deliveries | 11 across 15 missions (mean 0.73/run) |
| Funnel | 33 pick-ups → 11 zone arrivals → 11 deliveries |
| Transport conversion | 33% |
| Offload conversion | 100% — every robot that reached a zone delivered, no drop missed the boundary |
| Per-scenario means | 2.00, 0.00, 0.33, 0.67, 0.67 |
| Zero-delivery runs | 8 of 15 |
| Mean / peak final radiation | 599 / 2974 |

**The outcome is bimodal**: a run either delivers nothing or delivers one to three barrels.
Success tracks oscillation rather than distance — runs with at most one no-progress escape per
robot averaged 1.50 deliveries against 0.45 for the rest.

Full method, figures and analysis are in the project report.

---

## Reproducibility tools

Neither tool is needed at runtime; both document how a committed artefact was produced.

```bash
python3 solution/tools/generate_map.py          # regenerate map2.pgm from the world meshes (needs numpy)
```

```bash
python3 solution/tools/export_rosgraph.py rosgraph.png   # export the ROS graph while a scenario runs (needs rqt_graph, graphviz)
```
