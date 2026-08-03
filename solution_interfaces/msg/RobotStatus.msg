# Coordination status broadcast periodically by each robot_controller on the
# shared topic /robot_status. Used for decentralised target claiming and zone
# congestion avoidance.

string robot_id

# Value of the controller's finite state machine (see robot_controller.py State enum)
uint8 state

# Claimed barrel target (world frame). Other robots must not pursue a barrel
# within CLAIM_RADIUS of this position while the claim is active.
bool has_target
float64 target_x
float64 target_y

# Colour of the barrel currently being towed. 0=red, 1=blue, 255=not holding.
uint8 held_colour

# Collection zone the robot is currently heading to. 0=none, 1=zone A, 2=zone B.
uint8 target_zone

# Current estimated world-frame position of the robot, used for spatial
# dispersion (robots prefer barrels away from other robots).
float64 x
float64 y
