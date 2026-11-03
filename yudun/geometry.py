import numpy as np


def segment_hits_box(start, end, box):
    """Closed segment/AABB intersection, including parallel and tangent rays."""
    lower, upper = 0.0, 1.0
    for axis in range(2):
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-12:
            if start[axis] < box[axis] or start[axis] > box[axis + 2]:
                return False
        else:
            a = (box[axis] - start[axis]) / delta
            b = (box[axis + 2] - start[axis]) / delta
            lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
            if lower > upper:
                return False
    return upper > 1e-6 and lower < 1 - 1e-6


def is_visible(position, sensor_position, obstacles):
    return not any(segment_hits_box(sensor_position, position, box) for box in obstacles)


def ground_projection(pixel, intrinsics, world_from_camera, ground_z=0.0):
    """Undistorted pixel; camera optical axes x right/y down/z forward."""
    k, transform = np.asarray(intrinsics, float), np.asarray(world_from_camera, float)
    if k.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("K must be 3x3 and world_from_camera must be 4x4")
    if not np.isfinite(k).all() or not np.isfinite(transform).all():
        raise ValueError("non-finite calibration")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("extrinsic rotation must be a proper rotation")
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("invalid homogeneous transform")
    origin = transform[:3, 3]
    direction = rotation @ np.linalg.solve(k, [*pixel, 1])
    if abs(direction[2]) < 1e-9:
        raise ValueError("ray is parallel to ground")
    distance = (ground_z - origin[2]) / direction[2]
    if distance <= 0:
        raise ValueError("ground lies behind camera")
    return (origin + distance * direction)[:2]


def transform_points(points, world_from_sensor):
    points, transform = np.asarray(points, float), np.asarray(world_from_sensor, float)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError("expected Nx3 points and 4x4 transform")
    if not np.isfinite(points).all() or not np.isfinite(transform).all():
        raise ValueError("non-finite points or transform")
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("invalid homogeneous transform")
    return points @ transform[:3, :3].T + transform[:3, 3]
