# Empty arrays are valid sensor heartbeats.
uint16 schema_version 1
std_msgs/Header header
string sensor
yudun_interfaces/Detection3D[] detections
