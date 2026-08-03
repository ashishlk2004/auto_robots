#!/usr/bin/env python3
#
# AURO 2025 assessment solution - autonomous barrel collection controller.
#
# One instance of this node runs per robot (namespaced robotX). The controller
# implements a finite state machine that searches for barrels using the visual
# sensor, approaches and tows them, deposits them in a green collection zone,
# and manages robot contamination using the decontamination zone.
#
# Navigation is performed with Nav2 (AMCL localisation against a map generated
# from the world geometry, see solution/tools/generate_map.py). If Nav2 is not
# available (use_nav2:=false), the controller falls back to an internal
# waypoint-graph navigator that uses dead-reckoned odometry and reactive LiDAR
# avoidance.
#
# Multi-robot coordination is decentralised: robots share barrel sightings on
# /barrel_sightings and broadcast target claims on /robot_status, so that no
# two robots pursue the same barrel and collection zones are used evenly.

import heapq
import math
import random
import sys
from enum import IntEnum

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time

import tf2_ros

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

from auro_interfaces.srv import ItemRequest
from assessment_interfaces.msg import (Barrel, BarrelHolders, BarrelList,
                                       RadiationList, ZoneList)
from solution_interfaces.msg import BarrelSighting, RobotStatus


# ---------------------------------------------------------------------------
# World knowledge (constants of the assessment world, in the Gazebo/world
# frame, which coincides with the Nav2 map frame).
# ---------------------------------------------------------------------------

# Collection zones (centres of the 3x3 m green squares).
ZONE_A = (13.5, 9.4)
ZONE_B = (19.5, 9.4)
DECONTAMINATION_ZONE = (7.5, 9.4)
ZONE_HALF = 1.5

# Region east of the middle room in which delivered barrels accumulate:
# barrel sightings inside it are ignored so that already-collected barrels are
# not picked up again.
DELIVERED_REGION_X_MIN = 4.5
DELIVERED_REGION_Y_MIN = 7.0

# The single east-west corridor leading to the collection zones is the one
# shared resource in the arena. Two laden robots that meet inside it mark each
# other as obstacles and both stall, so laden robots reserve it one at a time
# (see corridor_reserved_by_peer). Robots wait west of the entrance instead.
CORRIDOR_X_MIN = 3.5
CORRIDOR_Y_MIN = 7.6
CORRIDOR_MAX_WAIT = 100.0

# Search vantage points: cluster 0 room first, then the west room (clusters
# 1-4 spawn in x [-15,-8], y [5,11] with static obstacles at (+-14,+-10)...).
SEARCH_POINTS = [
    (-2.0, 2.6),     # faces cluster 0 (centred at -2.0, 4.15)
    (-11.8, 8.0),    # centre of the west room, between the four obstacles
    (-15.2, 5.0),    # SW corner of the west room
    (-15.2, 11.2),   # NW corner
    (-8.9, 11.2),    # NE corner
    (-8.9, 5.0),     # SE corner
]

# Waypoint graph for the fallback (non-Nav2) navigator. Edges follow the
# corridors of the building; positions were validated against the generated
# occupancy map with an inflation of 0.30 m.
GRAPH_NODES = {
    'start':    (0.0, -1.2),
    'chL1':     (-2.4, 1.2),     # left channel of the middle room
    'chL2':     (-2.4, 6.6),
    'topL':     (-3.2, 7.8),     # junction above the middle room
    'chR1':     (1.6, 0.5),      # right channel of the middle room
    'chR2':     (1.6, 2.7),
    'chR3':     (2.5, 3.7),
    'chR4':     (2.6, 7.0),
    'topR':     (3.6, 8.5),
    'c0':       (0.1, 8.5),      # east-west corridor
    'c1':       (4.5, 8.5),
    'c2':       (9.0, 8.5),
    'c3':       (13.5, 8.5),
    'c4':       (17.0, 8.5),
    'c5':       (19.5, 8.5),
    'westdoor': (-6.5, 8.1),
    'westC':    (-11.8, 8.0),
    'westSW':   (-15.2, 5.0),
    'westNW':   (-15.2, 11.2),
    'westNE':   (-8.9, 11.2),
    'westSE':   (-8.9, 5.0),
    'cl0':      (-2.0, 2.6),
    'decE':     (7.5, 8.8),
    'zaE':      (13.5, 8.8),
    'zbE':      (19.5, 8.8),
}
GRAPH_EDGES = [
    ('start', 'chL1'), ('chL1', 'chL2'), ('chL2', 'topL'),
    ('start', 'chR1'), ('chR1', 'chR2'), ('chR2', 'chR3'),
    ('chR3', 'chR4'), ('chR4', 'topR'),
    ('chL1', 'cl0'), ('cl0', 'chL2'),
    ('topL', 'c0'), ('topL', 'westdoor'), ('westdoor', 'westC'),
    ('westC', 'westSW'), ('westC', 'westNW'), ('westC', 'westNE'),
    ('westC', 'westSE'), ('westSW', 'westNW'), ('westNE', 'westSE'),
    ('c0', 'topR'), ('c0', 'c1'), ('c1', 'c2'), ('c2', 'c3'),
    ('c3', 'c4'), ('c4', 'c5'), ('topR', 'c1'),
    ('c2', 'decE'), ('c3', 'zaE'), ('c5', 'zbE'),
]

# Camera model of the TurtleBot3 Waffle Pi simulated Pi camera (640x480,
# horizontal FOV 1.085595 rad).
IMG_W, IMG_H = 640, 480
FX = (IMG_W / 2.0) / math.tan(1.085595 / 2.0)   # ~530.5 px
CAMERA_HEIGHT = 0.14        # camera optical centre above the floor (approx)
BARREL_RADIUS = 0.15        # collision cylinder is 0.3 m diameter, 0.5 m tall
BARREL_CENTRE_Z = 0.25

# Pick-up geometry (see barrel_manager.filter_points_behind): the barrel must
# be within 0.45 m of the robot centre, inside a +-15 degree sector behind it.
PICKUP_MAX_DIST = 0.45
REVERSE_STOP_RANGE = 0.17   # rear LiDAR range at which to stop reversing
LIDAR_OFFSET_X = -0.064     # LiDAR position relative to base centre

# Sighting/claim bookkeeping.
SIGHTING_MERGE_RADIUS = 0.7
SIGHTING_MAX_AGE = 120.0
SIGHTING_MAX_RANGE = 7.0
CLAIM_RADIUS = 0.9

# Stuck detection. The twist watchdog only asks whether the wheels are
# turning, which cannot see a robot that is turning but not travelling:
# reactive avoidance oscillation (rotate away from an obstacle, rotate back
# towards the waypoint, repeat) holds |w| above the motion threshold
# indefinitely, so the robot is never declared stuck. The progress watchdog
# below instead requires real world-frame displacement.
PROGRESS_MIN_DIST = 0.35    # displacement that counts as making progress
PROGRESS_TIMEOUT = 25.0     # time allowed to cover it while under way

LINEAR_MAX = 0.24
ANGULAR_MAX = 1.0


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quaternion_from_yaw(yaw):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def angle_diff(a, b):
    """Smallest signed difference a-b, wrapped to [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


class State(IntEnum):
    INITIALISING = 0
    PLAN_NEXT = 1
    NAV_TO_SEARCH = 2
    SCANNING = 3
    NAV_TO_BARREL = 4
    APPROACH = 5
    GRAB = 6
    NAV_TO_ZONE = 7
    ENTER_ZONE = 8
    OFFLOAD = 9
    NAV_TO_DECON = 10
    DECONTAMINATE = 11
    ESCAPE = 12


class Sighting:
    __slots__ = ('x', 'y', 'colour', 'stamp', 'failed_attempts')

    def __init__(self, x, y, colour, stamp):
        self.x = x
        self.y = y
        self.colour = colour
        self.stamp = stamp
        self.failed_attempts = 0


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # --- Parameters -----------------------------------------------------
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('use_nav2', True)
        # Go to the decontamination zone once contamination >= this value
        # (contamination increases by 1 per second while towing a red barrel).
        self.declare_parameter('decon_threshold', 1)
        # Additional path-cost penalty (m) for choosing red barrels; can be
        # used to bias robots towards clean barrels in experiments.
        self.declare_parameter('red_penalty', 0.0)
        # Number of robots in the team, used to enable spatial partitioning.
        self.declare_parameter('num_robots', 1)

        self.initial_x = self.get_parameter('x').value
        self.initial_y = self.get_parameter('y').value
        self.initial_yaw = self.get_parameter('yaw').value
        self.use_nav2 = self.get_parameter('use_nav2').value
        self.decon_threshold = self.get_parameter('decon_threshold').value
        self.red_penalty = self.get_parameter('red_penalty').value
        self.num_robots = self.get_parameter('num_robots').value

        self.robot_id = self.get_namespace().strip('/')
        try:
            self.robot_number = int(''.join(filter(str.isdigit, self.robot_id)) or 1)
        except ValueError:
            self.robot_number = 1

        # Coordination is by target claiming plus a light dispersion penalty,
        # not by hard spatial partitioning. Assigning robots 2/3 exclusively to
        # the distant west room was measured to be counter-productive: they
        # spent whole missions travelling instead of collecting, while the
        # near cluster (10 barrels) went under-served. All robots therefore
        # share the full search route, but start it at different vantage points
        # so they spread out naturally.
        self.search_route = SEARCH_POINTS

        # --- Internal state -------------------------------------------------
        self.state = State.INITIALISING
        self.state_entered_time = None
        self.init_pose_publish_count = 0

        self.odom = None                  # latest nav_msgs/Odometry
        self.scan = None                  # latest sensor_msgs/LaserScan
        self.barrels_msg = None           # latest BarrelList
        self.barrels_stamp = None
        self.zones_msg = None
        self.holding = None               # colour of the held barrel, or None
        self.contamination = 0
        self.sightings = []               # shared barrel memory
        self.peer_status = {}             # robot_id -> RobotStatus

        self.target = None                # Sighting currently pursued
        # Stagger the search start so robots head to different vantage points.
        self.search_index = (self.robot_number - 1) % len(SEARCH_POINTS)
        self.search_sweeps = 0            # completed sweeps of the home route
        self.scan_accum = 0.0             # accumulated rotation while SCANNING
        self.last_scan_yaw = None
        self.grab_phase = 0
        self.grab_retries = 0
        self.align_started = 0.0
        self.offload_retries = 0
        self.decon_cooldown_until = 0.0
        self.redelivery = False           # re-offloading after a missed drop
        self.lane_switches = 0
        self.corridor_wait_started = None  # yielding at the corridor entrance
        self.zone_use_fallback = False    # deterministic route for this delivery
        self.decon_retries = 0
        self.nav_failures = 0
        self.zone_lane = 0                # lateral lane offset inside a zone
        self.approach_lost_time = None
        self.delivered_count = 0
        self.creep_target = None
        self.escape_plan = None
        self.after_escape = State.PLAN_NEXT
        self.grab_start_pose = None
        self.grab_barrel_pos = (0.0, 0.0)
        self.reverse_extra = 0.0
        self.pending_srv = None           # (name, future)
        self.mask_desired = False         # requested state of the LiDAR mask
        self.mask_confirmed = None        # last state acknowledged by the node
        self.mask_future = None
        self.mask_sent_value = None
        self.current_zone = ZONE_A
        self.post_offload_target = None
        self.lane_switch_time = 0.0

        # Stuck detection
        self.last_motion_time = None
        self.commanded_motion = False
        self.progress_ref = None          # (x, y, t) for the progress watchdog

        # Nav2 action bookkeeping
        self.nav_goal_handle = None
        self.nav_result = None            # None=idle/running, True/False done
        self.nav_send_future = None
        self.nav_result_future = None
        self.nav_started_time = None
        self.nav_goal_pose = None
        self.nav_reject_count = 0
        self.nav_resend_time = None
        # After several consecutive Nav2 failures the controller falls back
        # to its internal waypoint navigator for the next leg, which does not
        # depend on costmaps and can extract the robot from states that Nav2
        # considers to be in collision.
        self.consecutive_nav_failures = 0

        # Fallback navigator
        self.fallback_path = []           # list of (x, y) waypoints
        self.fallback_goal = None
        self.fallback_block_time = None

        # Dead-reckoned world pose (used when Nav2/AMCL is unavailable).
        self.dr_pose = (self.initial_x, self.initial_y, self.initial_yaw)
        self.odom_offset = None           # transform odom frame -> world frame

        # --- ROS interfaces -------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'initialpose', 10)
        self.status_pub = self.create_publisher(RobotStatus, '/robot_status', 10)
        self.sighting_pub = self.create_publisher(
            BarrelSighting, '/barrel_sightings', 10)

        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(BarrelList, 'barrels', self.barrels_callback, 10)
        self.create_subscription(ZoneList, 'zones', self.zones_callback, 10)
        self.create_subscription(BarrelHolders, '/barrel_holders',
                                 self.holders_callback, 10)
        self.create_subscription(RadiationList, '/radiation_levels',
                                 self.radiation_callback, 10)
        self.create_subscription(RobotStatus, '/robot_status',
                                 self.peer_status_callback, 10)
        self.create_subscription(BarrelSighting, '/barrel_sightings',
                                 self.peer_sighting_callback, 10)

        self.pick_up_client = self.create_client(ItemRequest, '/pick_up_item')
        self.offload_client = self.create_client(ItemRequest, '/offload_item')
        self.decon_client = self.create_client(ItemRequest, '/decontaminate')
        self.mask_param_client = self.create_client(
            SetParameters, 'dynamic_mask/set_parameters')
        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap, 'global_costmap/clear_entirely_global_costmap')
        self.clear_local_costmap_client = self.create_client(
            ClearEntireCostmap, 'local_costmap/clear_entirely_local_costmap')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.mask_timer = self.create_timer(1.0, self.mask_reconcile)

        self.get_logger().info(
            f"{self.robot_id} controller starting at "
            f"({self.initial_x:.2f}, {self.initial_y:.2f}, {self.initial_yaw:.2f}), "
            f"use_nav2={self.use_nav2}")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def odom_callback(self, msg):
        self.odom = msg
        # Fix the odom->world offset on first message using the known spawn pose.
        if self.odom_offset is None:
            oyaw = yaw_from_quaternion(msg.pose.pose.orientation)
            dyaw = self.initial_yaw - oyaw
            c, s = math.cos(dyaw), math.sin(dyaw)
            ox = self.initial_x - (c * msg.pose.pose.position.x
                                   - s * msg.pose.pose.position.y)
            oy = self.initial_y - (s * msg.pose.pose.position.x
                                   + c * msg.pose.pose.position.y)
            self.odom_offset = (ox, oy, dyaw)
        ox, oy, dyaw = self.odom_offset
        c, s = math.cos(dyaw), math.sin(dyaw)
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pyaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.dr_pose = (ox + c * px - s * py, oy + s * px + c * py, pyaw + dyaw)

        # Stuck detection based on measured twist.
        v = abs(msg.twist.twist.linear.x)
        w = abs(msg.twist.twist.angular.z)
        now = self.now_seconds()
        if v > 0.03 or w > 0.08 or not self.commanded_motion:
            self.last_motion_time = now

    def scan_callback(self, msg):
        self.scan = msg

    def barrels_callback(self, msg):
        self.barrels_msg = msg
        self.barrels_stamp = self.now_seconds()
        self.update_sightings_from_vision(msg)

    def zones_callback(self, msg):
        self.zones_msg = msg

    def holders_callback(self, msg):
        held = None
        for holder in msg.data:
            if holder.robot_id == self.robot_id:
                held = holder.colour
        self.holding = held

    def radiation_callback(self, msg):
        for radiation in msg.data:
            if radiation.robot_id == self.robot_id:
                self.contamination = radiation.level

    def peer_status_callback(self, msg):
        if msg.robot_id != self.robot_id:
            self.peer_status[msg.robot_id] = msg

    def peer_sighting_callback(self, msg):
        if msg.robot_id != self.robot_id:
            self.merge_sighting(msg.x, msg.y, msg.colour)

    # ------------------------------------------------------------------
    # Pose and perception helpers
    # ------------------------------------------------------------------

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def world_pose(self):
        """Best estimate of the robot pose in the world/map frame."""
        if self.use_nav2:
            try:
                t = self.tf_buffer.lookup_transform(
                    'map', 'base_footprint', Time())
                yaw = yaw_from_quaternion(t.transform.rotation)
                return (t.transform.translation.x,
                        t.transform.translation.y, yaw)
            except tf2_ros.TransformException:
                pass
        return self.dr_pose

    def sector_min(self, centre_deg, half_width_deg):
        """Minimum LiDAR range in a sector. Scan index 0 is straight ahead."""
        if self.scan is None:
            return float('inf')
        n = len(self.scan.ranges)
        if n == 0:
            return float('inf')
        best = float('inf')
        for offset in range(-half_width_deg, half_width_deg + 1):
            idx = int(round(centre_deg + offset)) % n
            r = self.scan.ranges[idx]
            if self.scan.range_min <= r <= self.scan.range_max:
                best = min(best, r)
        return best

    def front_clearance(self):
        return self.sector_min(0, 20)

    def rear_clearance(self):
        return self.sector_min(180, 8)

    def rear_target(self, max_range=1.2):
        """Nearest LiDAR return in the rear sector (+-45 deg around the back).
        Returns (angle_off_rear_rad, range) or None. A positive angle means
        the object lies counter-clockwise of straight-behind, and a positive
        angular velocity reduces it."""
        if self.scan is None:
            return None
        n = len(self.scan.ranges)
        if n == 0:
            return None
        best = None
        for off in range(-45, 46):
            idx = (180 + off) % n
            r = self.scan.ranges[idx]
            if self.scan.range_min <= r <= self.scan.range_max and \
                    r < max_range:
                if best is None or r < best[1]:
                    best = (math.radians(off), r)
        return best

    def barrel_bearing_rad(self, barrel):
        """Bearing of a detected barrel relative to the robot heading."""
        return -math.atan2(barrel.x - IMG_W / 2.0, FX)

    def barrel_distance(self, barrel):
        """Estimate distance to a barrel centre from vision (and LiDAR when
        close). Returns metres from robot centre."""
        d_area = float('inf')
        if barrel.size > 0:
            # Silhouette area ~ fx^2 * width * height / d^2
            d_area = FX * math.sqrt(0.85 * 2.0 * BARREL_RADIUS * 0.5
                                    / max(barrel.size, 1.0))
        # Refine with the LiDAR ray in the barrel's direction when nearby.
        bearing = self.barrel_bearing_rad(barrel)
        d_lidar = self.sector_min(int(round(math.degrees(bearing))), 4)
        if d_lidar < 2.5 and (d_area == float('inf')
                              or abs(d_lidar + BARREL_RADIUS - d_area) < 1.2):
            return d_lidar + BARREL_RADIUS + LIDAR_OFFSET_X
        return d_area

    def estimate_barrel_world(self, barrel):
        x, y, yaw = self.world_pose()
        d = self.barrel_distance(barrel)
        if not (0.2 < d < SIGHTING_MAX_RANGE):
            return None
        bearing = yaw + self.barrel_bearing_rad(barrel)
        return (x + d * math.cos(bearing), y + d * math.sin(bearing))

    def update_sightings_from_vision(self, msg):
        if self.state == State.INITIALISING:
            return
        for barrel in msg.data:
            pos = self.estimate_barrel_world(barrel)
            if pos is None:
                continue
            bx, by = pos
            # Ignore barrels in the delivery corridor / zones (already
            # collected) and positions outside the barrel spawning regions.
            if bx > DELIVERED_REGION_X_MIN and by > DELIVERED_REGION_Y_MIN:
                continue
            new = self.merge_sighting(bx, by, barrel.colour)
            if new:
                sighting = BarrelSighting()
                sighting.robot_id = self.robot_id
                sighting.colour = int(barrel.colour)
                sighting.x = bx
                sighting.y = by
                self.sighting_pub.publish(sighting)

    def merge_sighting(self, x, y, colour):
        """Insert or refresh a sighting. Returns True if it was new."""
        now = self.now_seconds()
        for s in self.sightings:
            if math.hypot(s.x - x, s.y - y) < SIGHTING_MERGE_RADIUS:
                s.x = 0.7 * s.x + 0.3 * x
                s.y = 0.7 * s.y + 0.3 * y
                s.colour = colour
                s.stamp = now
                return False
        self.sightings.append(Sighting(x, y, colour, now))
        return True

    def prune_sightings(self):
        now = self.now_seconds()
        self.sightings = [
            s for s in self.sightings
            if now - s.stamp < SIGHTING_MAX_AGE and s.failed_attempts < 3]

    def claimed_by_peer(self, sighting):
        """True if another robot has an active claim near this sighting."""
        for peer_id, status in self.peer_status.items():
            if not status.has_target:
                continue
            if math.hypot(status.target_x - sighting.x,
                          status.target_y - sighting.y) < CLAIM_RADIUS:
                try:
                    peer_number = int(''.join(filter(str.isdigit, peer_id)))
                except ValueError:
                    peer_number = 0
                # A peer that is already grabbing always wins; otherwise the
                # lower-numbered robot keeps the claim.
                if status.state in (int(State.APPROACH), int(State.GRAB)):
                    return True
                if peer_number < self.robot_number:
                    return True
        return False

    @staticmethod
    def in_corridor(x, y):
        """True if (x, y) lies in the shared corridor leading to the zones."""
        return x > CORRIDOR_X_MIN and y > CORRIDOR_Y_MIN

    def corridor_reserved_by_peer(self):
        """True if this robot must yield before entering the corridor: either
        another laden robot is already inside it, or a lower-numbered laden
        robot is also queuing to enter (the tie-break that prevents two robots
        from deadlocking while each waits for the other)."""
        for peer_id, status in self.peer_status.items():
            if status.held_colour == 255:      # peer is not carrying anything
                continue
            if self.in_corridor(status.x, status.y):
                return True
            if status.state == int(State.NAV_TO_ZONE):
                try:
                    peer_number = int(''.join(filter(str.isdigit, peer_id)))
                except ValueError:
                    peer_number = 0
                if peer_number < self.robot_number:
                    return True
        return False

    def choose_zone(self):
        """Pick a collection zone, preferring the assigned one but avoiding a
        zone that another robot is currently occupying. Robots 1/2 have fixed
        primaries; robot 3 alternates per delivery."""
        if self.robot_number >= 3:
            primary = ZONE_A if (self.delivered_count + self.robot_number) % 2 \
                else ZONE_B
        else:
            primary = ZONE_A if self.robot_number % 2 == 1 else ZONE_B
        secondary = ZONE_B if primary == ZONE_A else ZONE_A
        primary_id = 1 if primary == ZONE_A else 2
        for status in self.peer_status.values():
            if status.target_zone == primary_id and \
                    status.state in (int(State.ENTER_ZONE), int(State.OFFLOAD)):
                return secondary
        return primary

    # ------------------------------------------------------------------
    # Motion primitives
    # ------------------------------------------------------------------

    def drive(self, linear, angular):
        cmd = Twist()
        cmd.linear.x = max(-LINEAR_MAX, min(LINEAR_MAX, float(linear)))
        cmd.angular.z = max(-ANGULAR_MAX, min(ANGULAR_MAX, float(angular)))
        self.cmd_vel_pub.publish(cmd)
        self.commanded_motion = (abs(cmd.linear.x) > 0.01
                                 or abs(cmd.angular.z) > 0.05)

    def stop(self):
        self.drive(0.0, 0.0)
        self.commanded_motion = False

    def turn_towards(self, target_yaw, gain=1.5):
        """P-controlled in-place rotation. Returns True when aligned."""
        _, _, yaw = self.world_pose()
        err = angle_diff(target_yaw, yaw)
        if abs(err) < 0.05:
            self.stop()
            return True
        self.drive(0.0, max(-0.8, min(0.8, gain * err)))
        return False

    def creep_towards(self, tx, ty, speed=0.10, tolerance=0.10):
        """Slowly drive straight towards a nearby world point (used inside
        zones). Returns True on arrival."""
        x, y, yaw = self.world_pose()
        dist = math.hypot(tx - x, ty - y)
        if dist < tolerance:
            self.stop()
            return True
        target_yaw = math.atan2(ty - y, tx - x)
        err = angle_diff(target_yaw, yaw)
        if abs(err) > 0.3:
            self.drive(0.0, 1.2 * err)
        else:
            self.drive(min(speed, 0.5 * dist + 0.03), 1.0 * err)
        return False

    # ------------------------------------------------------------------
    # Nav2 interface
    # ------------------------------------------------------------------

    def nav_available(self):
        return self.use_nav2 and self.nav_client.server_is_ready()

    def start_navigation(self, x, y, yaw=None, force_fallback=False):
        """Begin navigating to a world pose using Nav2 or the fallback
        navigator. Set force_fallback to route via the deterministic waypoint
        graph regardless of Nav2 (used for the zone leg once Nav2 has proven
        unreliable, e.g. in multi-robot corridor congestion)."""
        self.cancel_navigation()
        self.stop()
        if yaw is None:
            px, py, _ = self.world_pose()
            yaw = math.atan2(y - py, x - px)
        self.nav_goal_pose = (x, y, yaw)
        self.nav_result = None
        self.nav_started_time = self.now_seconds()
        self.nav_reject_count = 0
        if self.nav_available() and self.consecutive_nav_failures < 3 \
                and not force_fallback:
            self.send_nav_goal()
        else:
            if self.nav_available():
                self.get_logger().warn(
                    f'{self.robot_id}: using the internal waypoint navigator '
                    'for this leg')
            self.fallback_plan(x, y)

    def send_nav_goal(self):
        x, y, yaw = self.nav_goal_pose
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation = quaternion_from_yaw(yaw)
        self.nav_send_future = self.nav_client.send_goal_async(goal)
        self.nav_send_future.add_done_callback(self.nav_goal_response)

    def nav_goal_response(self, future):
        handle = future.result()
        if handle is None or not handle.accepted:
            # Rejected goals are common while the Nav2 stack is still
            # activating: retry a few times with a short backoff.
            self.nav_reject_count += 1
            if self.nav_reject_count <= 5 and self.nav_goal_pose is not None:
                self.nav_resend_time = self.now_seconds() + 1.5
            else:
                self.nav_result = False
            return
        self.nav_goal_handle = handle
        self.nav_result_future = handle.get_result_async()
        self.nav_result_future.add_done_callback(self.nav_result_callback)

    def nav_result_callback(self, future):
        result = future.result()
        # status 4 = SUCCEEDED (action_msgs/GoalStatus)
        self.nav_result = (result is not None and result.status == 4)
        self.nav_goal_handle = None

    def cancel_navigation(self):
        if self.nav_goal_handle is not None:
            try:
                self.nav_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.nav_goal_handle = None
        self.nav_result = None
        self.nav_goal_pose = None
        self.nav_resend_time = None
        self.fallback_path = []
        self.fallback_goal = None

    def navigation_tick(self, timeout=180.0):
        """Advance navigation. Returns 'running', 'done' or 'failed'."""
        result = self.navigation_tick_inner(timeout)
        if result == 'done':
            self.consecutive_nav_failures = 0
        elif result == 'failed':
            self.consecutive_nav_failures += 1
        return result

    def navigation_tick_inner(self, timeout):
        if self.nav_goal_pose is None:
            return 'failed'
        gx, gy, _ = self.nav_goal_pose
        x, y, _ = self.world_pose()
        if self.nav_started_time is not None and \
                self.now_seconds() - self.nav_started_time > timeout:
            self.cancel_navigation()
            return 'failed'
        if self.nav_available() and not self.fallback_goal:
            if self.nav_resend_time is not None:
                if self.now_seconds() >= self.nav_resend_time:
                    self.nav_resend_time = None
                    self.send_nav_goal()
                return 'running'
            if self.nav_result is None:
                return 'running'
            done = self.nav_result
            if done:
                return 'done'
            # Nav2 reported failure; accept if we are close enough anyway.
            if math.hypot(gx - x, gy - y) < 0.4:
                return 'done'
            return 'failed'
        return self.fallback_tick()

    # ------------------------------------------------------------------
    # Fallback waypoint navigator (used when Nav2 is unavailable)
    # ------------------------------------------------------------------

    def fallback_plan(self, gx, gy):
        x, y, _ = self.world_pose()

        def nearest_node(px, py):
            return min(GRAPH_NODES,
                       key=lambda n: math.hypot(GRAPH_NODES[n][0] - px,
                                                GRAPH_NODES[n][1] - py))

        adj = {}
        for a, b in GRAPH_EDGES:
            pa, pb = GRAPH_NODES[a], GRAPH_NODES[b]
            w = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            adj.setdefault(a, []).append((b, w))
            adj.setdefault(b, []).append((a, w))

        start = nearest_node(x, y)
        goal = nearest_node(gx, gy)
        dist = {start: 0.0}
        prev = {}
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                break
            if d > dist.get(u, float('inf')):
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        path = [goal]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        path.reverse()
        self.fallback_path = [GRAPH_NODES[n] for n in path] + [(gx, gy)]
        self.fallback_goal = (gx, gy)
        self.fallback_block_time = None

    def fallback_tick(self):
        if not self.fallback_path:
            return 'failed'
        x, y, yaw = self.world_pose()
        tx, ty = self.fallback_path[0]
        final = len(self.fallback_path) == 1
        tolerance = 0.25 if final else 0.45
        if math.hypot(tx - x, ty - y) < tolerance:
            self.fallback_path.pop(0)
            if not self.fallback_path:
                self.stop()
                return 'done'
            return 'running'

        target_yaw = math.atan2(ty - y, tx - x)
        err = angle_diff(target_yaw, yaw)
        front = self.front_clearance()

        if front < 0.35:
            # Reactive avoidance: rotate towards the freer side. Threading out
            # of a barrel cluster legitimately takes a while, so allow longer
            # before declaring the route blocked.
            left = self.sector_min(45, 30)
            right = self.sector_min(-45 % 360, 30)
            self.drive(0.03, 0.6 if left > right else -0.6)
            if self.fallback_block_time is None:
                self.fallback_block_time = self.now_seconds()
            elif self.now_seconds() - self.fallback_block_time > 25.0:
                self.stop()
                return 'failed'
            return 'running'

        self.fallback_block_time = None
        if abs(err) > 0.5:
            self.drive(0.0, 1.2 * err)
        else:
            speed = 0.20 if front > 0.8 else 0.10
            self.drive(speed, 1.0 * err)
        return 'running'

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    def call_item_service(self, client, name):
        """Fire an ItemRequest service call (non-blocking)."""
        if not client.service_is_ready():
            return False
        request = ItemRequest.Request()
        request.robot_id = self.robot_id
        self.pending_srv = (name, client.call_async(request))
        return True

    def service_result(self, name):
        """Poll the pending service call. Returns response or None."""
        if self.pending_srv is None or self.pending_srv[0] != name:
            return None
        pending_name, future = self.pending_srv
        if not future.done():
            return None
        self.pending_srv = None
        try:
            return future.result()
        except Exception as e:
            self.get_logger().warn(f'service {pending_name} failed: {e}')
            return None

    def set_lidar_mask(self, enabled):
        """Request that the dynamic_mask node hide (or stop hiding) the rear
        LiDAR sector, so a towed barrel does not appear as an obstacle to
        Nav2. The actual service call is handled by mask_reconcile(), which
        retries until the node confirms the change."""
        self.mask_desired = enabled

    def mask_reconcile(self):
        """Keep the dynamic_mask node's parameters in sync with mask_desired.
        The masked sector is widened well beyond the barrel's ~49 degree
        silhouette because the scan is assembled while the robot rotates,
        which smears the barrel by ~10 degrees."""
        if self.mask_future is not None:
            if not self.mask_future.done():
                return
            ok = False
            try:
                result = self.mask_future.result()
                ok = all(r.successful for r in result.results)
            except Exception as e:
                self.get_logger().warn(f'mask parameter call failed: {e}')
            self.mask_future = None
            if ok:
                self.mask_confirmed = self.mask_sent_value
                self.get_logger().info(
                    f'{self.robot_id}: LiDAR mask '
                    f'{"enabled" if self.mask_confirmed else "disabled"}')
                # Purge obstacle marks made before the mask change engaged.
                self.clear_costmaps()
            return
        if self.mask_confirmed == self.mask_desired:
            return
        if not self.mask_param_client.service_is_ready():
            return
        request = SetParameters.Request()
        for name, value in (('ignore_sector_start', 125),
                            ('ignore_sector_end', 235)):
            parameter = Parameter()
            parameter.name = name
            parameter.value = ParameterValue(
                type=ParameterType.PARAMETER_INTEGER, integer_value=value)
            request.parameters.append(parameter)
        parameter = Parameter()
        parameter.name = 'mask_enabled'
        parameter.value = ParameterValue(
            type=ParameterType.PARAMETER_BOOL, bool_value=self.mask_desired)
        request.parameters.append(parameter)
        self.mask_sent_value = self.mask_desired
        self.mask_future = self.mask_param_client.call_async(request)

    def clear_costmaps(self):
        for client in (self.clear_global_costmap_client,
                       self.clear_local_costmap_client):
            if client.service_is_ready():
                client.call_async(ClearEntireCostmap.Request())

    # ------------------------------------------------------------------
    # Status publishing
    # ------------------------------------------------------------------

    def publish_status(self):
        msg = RobotStatus()
        msg.robot_id = self.robot_id
        msg.state = int(self.state)
        msg.has_target = self.target is not None
        if self.target is not None:
            msg.target_x = self.target.x
            msg.target_y = self.target.y
        x, y, _ = self.world_pose()
        msg.x = x
        msg.y = y
        msg.held_colour = int(self.holding) if self.holding is not None else 255
        if self.state in (State.NAV_TO_ZONE, State.ENTER_ZONE, State.OFFLOAD):
            msg.target_zone = 1 if self.current_zone == ZONE_A else 2
        else:
            msg.target_zone = 0
        self.status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Main control loop (10 Hz state machine)
    # ------------------------------------------------------------------

    def set_state(self, new_state):
        if new_state != self.state:
            self.get_logger().info(f'{self.robot_id}: {self.state.name} -> '
                                   f'{new_state.name}')
        self.state = new_state
        self.state_entered_time = self.now_seconds()

    def time_in_state(self):
        if self.state_entered_time is None:
            return 0.0
        return self.now_seconds() - self.state_entered_time

    def control_loop(self):
        handler = getattr(self, 'state_' + self.state.name.lower(), None)
        if handler is not None:
            handler()

        # Global stuck watchdog (only in states that drive autonomously).
        if self.state in (State.NAV_TO_SEARCH, State.NAV_TO_BARREL,
                          State.NAV_TO_ZONE, State.NAV_TO_DECON,
                          State.ENTER_ZONE):
            if self.commanded_motion and self.last_motion_time is not None \
                    and self.now_seconds() - self.last_motion_time > 8.0:
                self.get_logger().warn(f'{self.robot_id}: stuck, escaping')
                self.begin_escape(self.state)
            else:
                self.progress_watchdog()
        else:
            self.progress_ref = None

    def progress_watchdog(self):
        """Escape when the robot is being driven but is not getting anywhere.

        Complements the twist watchdog, which measures whether the wheels are
        turning and so cannot detect avoidance oscillation: rotating on the
        spot keeps |w| above the motion threshold for ever while net
        displacement stays at nil. Legitimate in-place turns (waypoint
        headings, re-alignment) last a few seconds, well inside the window.
        """
        if not self.commanded_motion:
            # Deliberately stopped, e.g. yielding at the corridor entrance.
            self.progress_ref = None
            return
        x, y, _ = self.world_pose()
        now = self.now_seconds()
        if self.progress_ref is None:
            self.progress_ref = (x, y, now)
            return
        rx, ry, since = self.progress_ref
        if math.hypot(x - rx, y - ry) > PROGRESS_MIN_DIST:
            self.progress_ref = (x, y, now)          # travelling, all is well
        elif now - since > PROGRESS_TIMEOUT:
            self.get_logger().warn(
                f'{self.robot_id}: no progress for {PROGRESS_TIMEOUT:.0f} s '
                f'({PROGRESS_MIN_DIST:.2f} m not covered), escaping')
            self.progress_ref = None
            self.begin_escape(self.state)

    # --- INITIALISING -------------------------------------------------

    def state_initialising(self):
        if self.odom is None or self.scan is None:
            return
        if self.state_entered_time is None:
            self.state_entered_time = self.now_seconds()

        if self.use_nav2:
            # Seed AMCL with the known spawn pose. Keep publishing until the
            # map->base transform appears, as AMCL may take a while to start.
            if int(self.time_in_state() * 2) > self.init_pose_publish_count:
                msg = PoseWithCovarianceStamped()
                msg.header.frame_id = 'map'
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.pose.pose.position.x = self.initial_x
                msg.pose.pose.position.y = self.initial_y
                msg.pose.pose.orientation = quaternion_from_yaw(self.initial_yaw)
                msg.pose.covariance[0] = 0.15
                msg.pose.covariance[7] = 0.15
                msg.pose.covariance[35] = 0.06
                self.initialpose_pub.publish(msg)
                self.init_pose_publish_count += 1

            if self.nav_client.server_is_ready():
                try:
                    self.tf_buffer.lookup_transform('map', 'base_footprint',
                                                    Time())
                    self.set_state(State.PLAN_NEXT)
                    return
                except tf2_ros.TransformException:
                    pass

            if self.time_in_state() > 45.0:
                self.get_logger().warn(
                    f'{self.robot_id}: Nav2 unavailable after 45 s, using '
                    'fallback navigator')
                self.use_nav2 = False
                self.set_state(State.PLAN_NEXT)
        else:
            if self.time_in_state() > 2.0:
                self.set_state(State.PLAN_NEXT)

    # --- PLAN_NEXT ----------------------------------------------------

    def state_plan_next(self):
        # After an offload, first creep clear of the dropped barrel so that
        # Nav2 does not start planning right next to a lethal obstacle.
        if self.post_offload_target is not None:
            if not self.creep_towards(*self.post_offload_target,
                                      speed=0.08, tolerance=0.12) and \
                    self.time_in_state() < 8.0:
                return
            self.post_offload_target = None

        self.stop()
        self.prune_sightings()
        self.target = None

        # Safety: if we are somehow holding a barrel, deliver it first.
        if self.holding is not None:
            self.current_zone = self.choose_zone()
            self.zone_lane = 0
            self.start_navigation(self.current_zone[0], 8.6, math.pi / 2,
                                  force_fallback=self.zone_use_fallback)
            self.set_state(State.NAV_TO_ZONE)
            return

        # Contamination management.
        if self.contamination >= self.decon_threshold and \
                self.now_seconds() >= self.decon_cooldown_until:
            self.start_navigation(DECONTAMINATION_ZONE[0], 8.6, math.pi / 2)
            self.set_state(State.NAV_TO_DECON)
            return

        # Choose the cheapest unclaimed sighting.
        best, best_cost = self.best_candidate()

        if best is not None:
            self.target = best
            sx, sy = self.standoff_point(best)
            self.start_navigation(sx, sy,
                                  math.atan2(best.y - sy, best.x - sx))
            self.set_state(State.NAV_TO_BARREL)
            return

        # Nothing known: continue this robot's search route. If its own region
        # is exhausted (route fully swept with nothing found), fall back to the
        # full arena route so barrels are never left uncollected.
        route = self.search_route
        if self.search_sweeps >= 2 and route is not SEARCH_POINTS:
            route = SEARCH_POINTS
        wx, wy = route[self.search_index % len(route)]
        self.search_index += 1
        if self.search_index % len(route) == 0:
            self.search_sweeps += 1
        self.start_navigation(wx, wy)
        self.set_state(State.NAV_TO_SEARCH)

    def best_candidate(self):
        """Cheapest unclaimed sighting, by travel distance. Barrels already
        claimed by a peer are excluded outright; a light pairwise penalty then
        nudges robots towards different barrels so the team spreads out without
        anyone being sent on a long detour."""
        x, y, _ = self.world_pose()
        best, best_cost = None, float('inf')
        for s in self.sightings:
            if self.claimed_by_peer(s):
                continue
            cost = math.hypot(s.x - x, s.y - y)
            if s.colour == Barrel.RED:
                cost += self.red_penalty
            # Light pairwise dispersion for residual ties.
            for status in self.peer_status.values():
                if status.has_target and \
                        math.hypot(status.target_x - s.x,
                                   status.target_y - s.y) < 1.5:
                    cost += 1.5
                if math.hypot(status.x - s.x, status.y - s.y) < 2.0:
                    cost += 1.0
            if cost < best_cost:
                best, best_cost = s, cost
        return best, best_cost

    def standoff_point(self, sighting):
        """A pose about 0.9 m short of the barrel, approached from the robot's
        current direction."""
        x, y, _ = self.world_pose()
        dx, dy = x - sighting.x, y - sighting.y
        d = math.hypot(dx, dy)
        if d < 0.1:
            return (x, y)
        f = 0.9 / d
        return (sighting.x + dx * f, sighting.y + dy * f)

    # --- NAV_TO_SEARCH ------------------------------------------------

    def state_nav_to_search(self):
        # Opportunistic: if a cheap unclaimed barrel is already known (taking
        # the dispersion penalties into account, so we do not pile onto a
        # cluster other robots are working), go for it instead of finishing
        # the search leg.
        _, cost = self.best_candidate()
        if cost < 3.5:
            self.cancel_navigation()
            self.set_state(State.PLAN_NEXT)
            return

        result = self.navigation_tick(timeout=150.0)
        if result == 'done':
            self.nav_failures = 0
            self.scan_accum = 0.0
            self.last_scan_yaw = None
            self.set_state(State.SCANNING)
        elif result == 'failed':
            self.nav_failures += 1
            if self.nav_failures >= 2:
                self.nav_failures = 0
                self.begin_escape(State.PLAN_NEXT)
            else:
                self.set_state(State.PLAN_NEXT)

    # --- SCANNING -----------------------------------------------------

    def state_scanning(self):
        _, _, yaw = self.world_pose()
        if self.last_scan_yaw is not None:
            self.scan_accum += abs(angle_diff(yaw, self.last_scan_yaw))
        self.last_scan_yaw = yaw
        if self.scan_accum >= 2.0 * math.pi or self.time_in_state() > 20.0:
            self.stop()
            self.set_state(State.PLAN_NEXT)
            return
        self.drive(0.0, 0.6)

    # --- NAV_TO_BARREL ------------------------------------------------

    def state_nav_to_barrel(self):
        if self.target is None:
            self.cancel_navigation()
            self.set_state(State.PLAN_NEXT)
            return
        result = self.navigation_tick(timeout=150.0)
        if result == 'done':
            self.nav_failures = 0
            self.approach_lost_time = None
            self.set_state(State.APPROACH)
        elif result == 'failed':
            self.target.failed_attempts += 1
            self.nav_failures += 1
            if self.nav_failures >= 2:
                self.nav_failures = 0
                self.begin_escape(State.PLAN_NEXT)
            else:
                self.set_state(State.PLAN_NEXT)

    # --- APPROACH -----------------------------------------------------

    def visible_target_barrel(self):
        """The detected barrel that best matches the current target claim."""
        if self.barrels_msg is None or self.barrels_stamp is None:
            return None
        if self.now_seconds() - self.barrels_stamp > 1.0:
            return None
        best, best_d = None, float('inf')
        for barrel in self.barrels_msg.data:
            pos = self.estimate_barrel_world(barrel)
            if pos is None:
                # Extremely close barrels can defeat the estimator; accept
                # them based on bearing alone.
                if self.target is not None and \
                        abs(self.barrel_bearing_rad(barrel)) < 0.6:
                    d = 0.0
                    if d < best_d:
                        best, best_d = barrel, d
                continue
            if self.target is None:
                d = self.barrel_distance(barrel)
            else:
                d = math.hypot(pos[0] - self.target.x, pos[1] - self.target.y)
                if d > 2.0:
                    continue
            if d < best_d:
                best, best_d = barrel, d
        return best

    def state_approach(self):
        barrel = self.visible_target_barrel()
        now = self.now_seconds()

        if barrel is None:
            if self.approach_lost_time is None:
                self.approach_lost_time = now
            lost_for = now - self.approach_lost_time
            if lost_for < 6.0:
                # Sweep in place to re-acquire.
                self.drive(0.0, 0.45 if int(lost_for / 2.0) % 2 == 0 else -0.45)
                return
            if self.target is not None:
                self.target.failed_attempts += 1
            self.stop()
            self.set_state(State.PLAN_NEXT)
            return

        self.approach_lost_time = None
        bearing = self.barrel_bearing_rad(barrel)
        distance = self.barrel_distance(barrel)

        # Keep the shared memory up to date while we close in.
        if self.target is not None:
            pos = self.estimate_barrel_world(barrel)
            if pos is not None and distance < 3.0:
                self.target.x, self.target.y = pos
                self.target.stamp = now
                self.target.colour = barrel.colour

        if distance <= 0.55:
            self.stop()
            self.grab_phase = 0
            self.grab_retries = 0
            self.reverse_extra = 0.0
            x, y, yaw = self.world_pose()
            self.grab_barrel_pos = (x + distance * math.cos(yaw + bearing),
                                    y + distance * math.sin(yaw + bearing))
            self.set_state(State.GRAB)
            return

        # Guard against non-target obstacles dead ahead: hand back to Nav2
        # for a fresh standoff rather than blindly manoeuvring.
        front = self.front_clearance()
        if front < 0.30 and distance > 0.9:
            self.stop()
            if self.target is not None:
                sx, sy = self.standoff_point(self.target)
                self.start_navigation(
                    sx, sy, math.atan2(self.target.y - sy, self.target.x - sx))
                self.set_state(State.NAV_TO_BARREL)
            else:
                self.set_state(State.PLAN_NEXT)
            return

        angular = 2.0 * bearing
        if abs(bearing) > 0.35:
            linear = 0.0
        elif distance > 1.5:
            linear = 0.20
        else:
            linear = 0.08 + 0.08 * (distance - 0.55)
        self.drive(linear, angular)

        if self.time_in_state() > 60.0:
            if self.target is not None:
                self.target.failed_attempts += 1
            self.set_state(State.PLAN_NEXT)

    # --- GRAB ---------------------------------------------------------
    #
    # Docking sequence, closed loop on the rear LiDAR so that it tolerates
    # localisation error and approach misalignment:
    #   phase 0 TURN    - coarse 180 degree turn away from the barrel
    #   phase 1 ALIGN   - centre the nearest rear LiDAR return behind us
    #   phase 2 REVERSE - back up towards it, steering to keep it centred
    #   phase 3 CALL    - request /pick_up_item
    #   phase 4 WAIT    - process the response, retrying closer if needed

    def state_grab(self):
        if self.grab_phase == 0:
            bx, by = self.grab_barrel_pos
            x, y, _ = self.world_pose()
            away_yaw = math.atan2(y - by, x - bx)
            if self.turn_towards(away_yaw) or self.time_in_state() > 15.0:
                self.grab_phase = 1
                self.align_started = self.now_seconds()
            return

        if self.grab_phase == 1:
            target = self.rear_target()
            if target is None:
                self.drive(0.0, 0.0)
                if self.now_seconds() - self.align_started > 4.0:
                    self.abort_grab()
                return
            off, _ = target
            if abs(off) < 0.05:
                self.stop()
                self.grab_phase = 2
                self.grab_start_pose = self.world_pose()
                return
            self.drive(0.0, max(-0.5, min(0.5, 1.5 * off)))
            return

        if self.grab_phase == 2:
            target = self.rear_target()
            x, y, _ = self.world_pose()
            sx, sy, _ = self.grab_start_pose
            reversed_dist = math.hypot(x - sx, y - sy)
            stop_range = max(0.13, REVERSE_STOP_RANGE - 0.02 * self.grab_retries)
            if target is None:
                # Lost it (rolled away or now below the LiDAR minimum range):
                # if we have already backed up a fair way, just try the pick
                # up; otherwise re-align.
                self.stop()
                if reversed_dist > 0.15:
                    self.grab_phase = 3
                else:
                    self.grab_phase = 1
                    self.align_started = self.now_seconds()
                return
            off, r = target
            if r <= stop_range or reversed_dist > 0.85:
                self.stop()
                self.grab_phase = 3
                return
            self.drive(-0.06, max(-0.4, min(0.4, 0.8 * off)))
            return

        if self.grab_phase == 3:
            if self.call_item_service(self.pick_up_client, 'pick_up'):
                self.grab_phase = 4
            elif self.time_in_state() > 40.0:
                self.abort_grab()
            return

        if self.grab_phase == 4:
            response = self.service_result('pick_up')
            if response is None:
                if self.time_in_state() > 60.0:
                    self.abort_grab()
                return
            if response.success or 'already holding' in response.message:
                self.get_logger().info(
                    f'{self.robot_id}: picked up barrel ({response.message})')
                self.on_barrel_secured()
                return
            self.grab_retries += 1
            self.get_logger().info(
                f'{self.robot_id}: pick up failed ({response.message}), '
                f'retry {self.grab_retries}')
            if self.grab_retries <= 3:
                # Re-align on the rear target and back up a little closer.
                self.grab_phase = 1
                self.align_started = self.now_seconds()
            else:
                self.abort_grab()

    def abort_grab(self):
        self.stop()
        if self.target is not None:
            self.target.failed_attempts += 1
        self.pending_srv = None
        # Pull forward away from the barrel before replanning, so that Nav2
        # does not start wedged against a lethal obstacle.
        x, y, yaw = self.world_pose()
        self.post_offload_target = (x + 0.4 * math.cos(yaw),
                                    y + 0.4 * math.sin(yaw))
        self.set_state(State.PLAN_NEXT)

    def on_barrel_secured(self):
        self.set_lidar_mask(True)
        if self.redelivery:
            # We re-grabbed a barrel that missed the zone: creep straight back
            # in (creep_target was already moved to a different lane) rather
            # than re-doing the whole navigation.
            self.redelivery = False
            self.set_state(State.ENTER_ZONE)
            return
        # Remove the collected barrel from the shared memory.
        bx, by = self.grab_barrel_pos
        self.sightings = [
            s for s in self.sightings
            if math.hypot(s.x - bx, s.y - by) > SIGHTING_MERGE_RADIUS]
        self.target = None
        self.offload_retries = 0
        self.lane_switches = 0
        self.current_zone = self.choose_zone()
        # Rotate the starting lane so successive drops spread across the zone
        # instead of stacking up and blocking the entrance.
        self.zone_lane = (0, -1, 1)[self.delivered_count % 3]
        # The robot has just docked inside a barrel cluster, so the surrounding
        # barrels fill the global costmap with lethal cells and Nav2 cannot
        # produce a plan from here at all. Leave on the deterministic waypoint
        # route, which ignores costmaps and drives out of the cluster towards
        # the corridor; NAV_TO_ZONE hands back to Nav2 once clear of the room.
        self.zone_use_fallback = True
        self.start_navigation(self.current_zone[0], 8.6, math.pi / 2,
                              force_fallback=True)
        self.set_state(State.NAV_TO_ZONE)

    # --- NAV_TO_ZONE --------------------------------------------------

    def state_nav_to_zone(self):
        # Corridor reservation: hold west of the entrance while another laden
        # robot is using it, rather than entering and deadlocking with it.
        # Waiting does not count towards the navigation timeout, and is capped
        # so a lost peer cannot block this robot indefinitely.
        if self.num_robots > 1 and self.holding is not None:
            x, y, _ = self.world_pose()
            if not self.in_corridor(x, y) and self.corridor_reserved_by_peer():
                if self.corridor_wait_started is None:
                    self.corridor_wait_started = self.now_seconds()
                    self.get_logger().info(
                        f'{self.robot_id}: yielding, corridor in use')
                waited = self.now_seconds() - self.corridor_wait_started
                if waited < CORRIDOR_MAX_WAIT:
                    self.stop()
                    if self.nav_started_time is not None:
                        self.nav_started_time += 0.1   # pause the timeout
                    return
            elif self.corridor_wait_started is not None:
                self.corridor_wait_started = None

        # The leg starts on the deterministic waypoint route (see
        # on_barrel_secured). Keep the timeout short so a blocked delivery
        # escalates quickly instead of burning the whole mission.
        result = self.navigation_tick(timeout=150.0)
        if result == 'done':
            self.nav_failures = 0
            lane_x = self.current_zone[0] + self.zone_lane * 0.9
            self.creep_target = (lane_x, 9.55)
            self.set_state(State.ENTER_ZONE)
        elif result == 'failed':
            self.nav_failures += 1
            if self.nav_failures == 1:
                # Re-plan the waypoint route from the current position.
                self.start_navigation(self.current_zone[0], 8.6, math.pi / 2,
                                      force_fallback=True)
            elif self.nav_failures == 2:
                # Still blocked: try the other zone via the waypoint route.
                self.current_zone = (ZONE_B if self.current_zone == ZONE_A
                                     else ZONE_A)
                self.start_navigation(self.current_zone[0], 8.6, math.pi / 2,
                                      force_fallback=True)
            else:
                self.nav_failures = 0
                self.begin_escape(State.PLAN_NEXT)

    # --- ENTER_ZONE ---------------------------------------------------

    def state_enter_zone(self):
        tx, ty = self.creep_target
        x, y, _ = self.world_pose()

        # If the lane ahead is blocked by already-delivered barrels, only
        # offload in place when we are deep enough that the towed barrel will
        # land well inside the zone even with localisation error; otherwise
        # shift to a neighbouring lane and try there.
        front = self.front_clearance()
        if front < 0.35:
            if y >= 9.0 or self.lane_switches >= 3:
                self.stop()
                self.set_state(State.OFFLOAD)
                return
            # Blocked near the zone entrance: try a neighbouring lane
            # (at most once every 2 seconds to avoid flapping).
            if self.now_seconds() - self.lane_switch_time > 2.0:
                self.lane_switch_time = self.now_seconds()
                self.lane_switches += 1
                if self.zone_lane == 0:
                    self.zone_lane = -1 if self.robot_number % 2 == 1 else 1
                else:
                    self.zone_lane = -self.zone_lane
                lane_x = self.current_zone[0] + self.zone_lane * 0.9
                self.creep_target = (lane_x, 9.55)
            self.drive(-0.08, 0.0)
            return

        if self.creep_towards(tx, ty, speed=0.10, tolerance=0.12):
            self.set_state(State.OFFLOAD)
            return

        if self.time_in_state() > 45.0:
            self.set_state(State.OFFLOAD)

    # --- OFFLOAD ------------------------------------------------------

    def state_offload(self):
        response = self.service_result('offload')
        if response is None:
            if self.pending_srv is None:
                if not self.call_item_service(self.offload_client, 'offload'):
                    if self.time_in_state() > 30.0:
                        self.set_state(State.PLAN_NEXT)
            elif self.time_in_state() > 45.0:
                self.pending_srv = None
                self.set_state(State.PLAN_NEXT)
            return

        if response.success and 'collection zone' in response.message:
            self.delivered_count += 1
            self.get_logger().info(
                f'{self.robot_id}: delivered barrel #{self.delivered_count} '
                f'({response.message})')
            self.set_lidar_mask(False)
            self.leave_zone_and_continue()
            return

        if response.success:
            # Offloaded but not inside a zone: pick it straight back up (it is
            # right behind us), creep deeper and retry.
            self.offload_retries += 1
            self.get_logger().warn(
                f'{self.robot_id}: offload missed the zone '
                f'(attempt {self.offload_retries}): {response.message}')
            if self.offload_retries <= 3:
                # The barrel is directly behind us: request the pick up
                # straight away (GRAB phase 3 = CALL), then creep back in on
                # a different lane and deeper before dropping again.
                self.redelivery = True
                self.grab_phase = 3
                self.grab_retries = 0
                x, y, yaw = self.world_pose()
                self.grab_barrel_pos = (x - 0.38 * math.cos(yaw),
                                        y - 0.38 * math.sin(yaw))
                if self.zone_lane == 0:
                    self.zone_lane = -1 if self.robot_number % 2 == 1 else 1
                else:
                    self.zone_lane = -self.zone_lane
                lane_x = self.current_zone[0] + self.zone_lane * 0.9
                self.creep_target = (lane_x, 9.7)
                self.set_state(State.GRAB)
                return
            self.set_lidar_mask(False)
            self.leave_zone_and_continue()
            return

        # response.success is False (e.g. not holding an item).
        self.get_logger().warn(
            f'{self.robot_id}: offload failed: {response.message}')
        self.set_lidar_mask(False)
        self.set_state(State.PLAN_NEXT)

    def leave_zone_and_continue(self):
        # Creep forward clear of the dropped barrel before replanning.
        x, y, yaw = self.world_pose()
        self.post_offload_target = (x + 0.3 * math.cos(yaw),
                                    y + 0.3 * math.sin(yaw))
        self.set_state(State.PLAN_NEXT)

    # --- NAV_TO_DECON / DECONTAMINATE ---------------------------------

    def state_nav_to_decon(self):
        result = self.navigation_tick(timeout=240.0)
        if result == 'done':
            self.nav_failures = 0
            # Stop just inside the zone boundary (y >= 7.9): the shower
            # structure occupies the middle of the zone from y ~ 9.4, so
            # creeping deeper would collide with it.
            self.creep_target = (DECONTAMINATION_ZONE[0], 9.0)
            self.decon_retries = 0
            self.set_state(State.DECONTAMINATE)
        elif result == 'failed':
            self.nav_failures += 1
            if self.nav_failures <= 2:
                self.start_navigation(DECONTAMINATION_ZONE[0], 8.6,
                                      math.pi / 2)
            else:
                self.nav_failures = 0
                self.begin_escape(State.PLAN_NEXT)

    def state_decontaminate(self):
        # First creep to the middle of the cyan zone, then request the wash.
        if self.creep_target is not None:
            if not self.creep_towards(*self.creep_target, speed=0.10,
                                      tolerance=0.15):
                if self.time_in_state() > 40.0:
                    self.creep_target = None
                return
            self.creep_target = None

        response = self.service_result('decontaminate')
        if response is None:
            if self.pending_srv is None:
                self.call_item_service(self.decon_client, 'decontaminate')
            elif self.time_in_state() > 60.0:
                self.pending_srv = None
                self.set_state(State.PLAN_NEXT)
            return

        if response.success:
            self.get_logger().info(f'{self.robot_id}: decontaminated')
            # Update the cached level immediately: the /radiation_levels topic
            # is only published at 1 Hz and a stale value would send us
            # straight back to the decontamination zone.
            self.contamination = 0
            self.set_state(State.PLAN_NEXT)
        else:
            self.decon_retries += 1
            # Retry at different spots inside the zone, staying south of the
            # shower structure.
            retry_targets = [(7.1, 9.0), (7.9, 9.0), (7.5, 8.6)]
            if self.decon_retries <= len(retry_targets):
                self.creep_target = retry_targets[self.decon_retries - 1]
            else:
                self.get_logger().warn(
                    f'{self.robot_id}: giving up on decontamination for now: '
                    f'{response.message}')
                # Do not starve collection: retry after a cooldown.
                self.decon_cooldown_until = self.now_seconds() + 120.0
                self.set_state(State.PLAN_NEXT)

    # --- ESCAPE (recovery behaviour) ----------------------------------

    def most_open_direction(self):
        """Bearing (radians, robot frame) of the widest free direction
        according to the LiDAR, considering +-25 degree windows."""
        if self.scan is None:
            return math.pi  # default: back away
        n = len(self.scan.ranges)
        if n == 0:
            return math.pi
        ranges = []
        for r in self.scan.ranges:
            if self.scan.range_min <= r <= self.scan.range_max:
                ranges.append(min(r, 2.5))
            else:
                ranges.append(2.5)   # no return = clear
        best_deg, best_score = 0, -1.0
        for centre in range(0, 360, 15):
            window = [ranges[(centre + o) % n] for o in range(-25, 26)]
            score = min(window)  # conservative: worst ray in the window
            if score > best_score:
                best_score, best_deg = score, centre
        if best_deg > 180:
            best_deg -= 360
        return math.radians(best_deg)

    def begin_escape(self, next_state):
        """Recovery: rotate towards the most open direction seen by the LiDAR
        and drive out of the pocket, then resume."""
        self.cancel_navigation()
        self.pending_srv = None
        self.after_escape = next_state
        # Phantom obstacle marks (e.g. smeared by a towed barrel during
        # rotations) are a common cause of being walled in: purge them.
        self.clear_costmaps()
        x, y, yaw = self.world_pose()
        # Add a little randomness so repeated escapes do not repeat the exact
        # same manoeuvre.
        offset = self.most_open_direction() + random.uniform(-0.2, 0.2)
        self.escape_plan = {
            'phase': 0,
            'start': (x, y),
            'target_yaw': yaw + offset,
            'returns': 0,
        }
        self.set_state(State.ESCAPE)

    def state_escape(self):
        plan = self.escape_plan
        if plan is None:
            self.set_state(self.after_escape)
            return
        x, y, yaw = self.world_pose()
        # Keep the escape a short nudge (~0.5 m): just enough to clear the
        # immediate cluster so Nav2 can re-plan. Driving further chases the
        # widest LiDAR opening, which in the middle room is the dead-end spawn
        # area rather than the narrow exit channel, and strands a towed barrel.
        goal_dist = 0.5
        if plan['phase'] == 0:
            # Turn gently while towing so the barrel stays inside the masked
            # LiDAR sector.
            gain = 0.7 if self.holding is not None else 1.5
            if self.turn_towards(plan['target_yaw'], gain=gain) or \
                    self.time_in_state() > 12.0 + 8.0 * plan['returns']:
                self.stop()
                plan['phase'] = 1
            return
        if plan['phase'] == 1:
            moved = math.hypot(x - plan['start'][0], y - plan['start'][1])
            if moved > goal_dist or self.time_in_state() > 40.0:
                self.stop()
                self.escape_plan = None
                self.set_state(self.after_escape)
                return
            if self.front_clearance() < 0.28:
                # Blocked: pick a fresh open direction (up to 3 times).
                self.stop()
                plan['returns'] += 1
                if plan['returns'] > 3:
                    self.escape_plan = None
                    self.set_state(self.after_escape)
                    return
                plan['target_yaw'] = yaw + self.most_open_direction() \
                    + random.uniform(-0.15, 0.15)
                plan['phase'] = 0
                return
            self.drive(0.10, 0.0)

    def destroy_node(self):
        try:
            self.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    node = RobotController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
