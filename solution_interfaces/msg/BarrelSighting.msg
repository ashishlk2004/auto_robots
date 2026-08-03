# Estimated world-frame position of a barrel observed by a robot's visual
# sensor. Broadcast on the shared topic /barrel_sightings so that all robots
# maintain a common memory of where barrels are.

string robot_id

# 0=red (contaminated), 1=blue (clean). Matches assessment_interfaces/Barrel.
uint8 colour

float64 x
float64 y
