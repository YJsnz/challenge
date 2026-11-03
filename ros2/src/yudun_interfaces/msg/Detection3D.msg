# Map frame, metres, row-major 2x2 position covariance.
uint16 schema_version 1
std_msgs/Header header
string observation_id
string category
float64 confidence
float64 quality
float64 x
float64 y
float64[4] covariance
float64 radius
