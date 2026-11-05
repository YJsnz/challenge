"""Map/ground camera pose fitting with explicitly independent ground checkpoints."""
import numpy as np
from ..geometry import ground_projection, transform_points
from .scene import points, rigid


def camera_parameters(data):
    k = np.asarray(data['intrinsics'],float)
    if (k.shape != (3,3) or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0
            or not np.allclose(k[2],[0,0,1]) or abs(k[0,1])+abs(k[1,0]) > 1e-9):
        raise ValueError("intrinsics require positive focal lengths, zero skew and last row [0,0,1]")
    distortion = np.asarray(data.get('distortion',[]),float).reshape(-1)
    if len(distortion) not in (0,4,5,8,12,14) or not np.isfinite(distortion).all():
        raise ValueError("unsupported or non-finite OpenCV distortion coefficients")
    size = [data['image_width'],data['image_height']]
    if any(not isinstance(v,int) or isinstance(v,bool) or v <= 0 for v in size):
        raise ValueError("camera image width/height must be positive integers")
    return k,distortion


def correspondences(data, camera, minimum):
    world = points(data['map_points_m'],minimum=minimum)
    pixels = points(data['image_points_px'],2,minimum)
    if len(world) != len(pixels):
        raise ValueError("map and pixel correspondence counts differ")
    if np.any(pixels < 0) or np.any(pixels >= [camera['image_width'],camera['image_height']]):
        raise ValueError("control pixels lie outside the calibrated image")
    return world,pixels


def independent(fit,check):
    if any(np.any(np.linalg.norm(fit-p,axis=1) < 1e-7) for p in check):
        raise ValueError("checkpoints must be spatially independent of fitting control points")


def _reproject(scene, camera, world):
    import cv2
    k,distortion = camera_parameters(camera)
    map_from_camera = scene.map_from_ground@rigid(camera['world_from_camera'])
    camera_from_map = np.linalg.inv(map_from_camera)
    if np.any(transform_points(world,camera_from_map)[:,2] <= 1e-6):
        raise ValueError("map control points must lie in front of the camera")
    rvec,_ = cv2.Rodrigues(camera_from_map[:3,:3])
    pixels,_ = cv2.projectPoints(world,rvec,camera_from_map[:3,3],k,distortion)
    return pixels.reshape(-1,2)


def evaluate_camera(scene, camera, check_data):
    import cv2
    k,distortion = camera_parameters(camera)
    world,pixels = correspondences(check_data,camera,3)
    ground = scene.to_ground(world)
    ground_offset = np.abs(ground[:,2])
    undistorted = cv2.undistortPoints(pixels.reshape(-1,1,2),k,distortion,P=k).reshape(-1,2)
    estimated = np.array([ground_projection(p,k,camera['world_from_camera']) for p in undistorted])
    map_estimated = scene.lift(estimated)
    position_errors = np.linalg.norm(map_estimated-world,axis=1)
    pixel_errors = np.linalg.norm(_reproject(scene,camera,world)-pixels,axis=1)
    return {'checkpoints':len(world),'position_errors_m':position_errors.tolist(),
            'position_rmse_m':float(np.sqrt(np.mean(position_errors**2))),
            'position_max_m':float(position_errors.max()),
            'ground_plane_offset_max_m':float(ground_offset.max()),
            'reprojection_errors_px':pixel_errors.tolist(),
            'reprojection_rmse_px':float(np.sqrt(np.mean(pixel_errors**2))),
            'reprojection_max_px':float(pixel_errors.max()),
            'map_estimated_m':map_estimated.tolist(),'map_reference_m':world.tolist()}


def calibrate_camera(scene, data):
    import cv2
    if not scene.registered:
        raise ValueError("register the map ground frame before calibrating a camera")
    k,distortion = camera_parameters(data)
    world,pixels = correspondences(data['control_points'],data,4)
    check_world,_ = correspondences(data['check_points'],data,3)
    independent(world,check_world)
    rank = np.linalg.matrix_rank(world-world.mean(axis=0),tol=1e-8)
    if rank < 2 or np.linalg.matrix_rank(pixels-pixels.mean(axis=0),tol=1e-8) < 2:
        raise ValueError("camera control points must not be collinear")
    method = cv2.SOLVEPNP_IPPE if rank == 2 else cv2.SOLVEPNP_SQPNP
    solved = cv2.solvePnPGeneric(world,pixels,k,distortion,flags=method)
    choices = []
    if solved[0]:
        for rvec,tvec in zip(solved[1],solved[2]):
            rvec,tvec = cv2.solvePnPRefineLM(world,pixels,k,distortion,rvec.copy(),tvec.copy())
            rotation,_ = cv2.Rodrigues(rvec)
            camera_from_map = np.eye(4)
            camera_from_map[:3,:3],camera_from_map[:3,3] = rotation,tvec.reshape(3)
            ground_from_camera = np.linalg.inv(scene.map_from_ground)@np.linalg.inv(camera_from_map)
            if ground_from_camera[2,3] <= 0 or np.any(transform_points(world,camera_from_map)[:,2] <= 1e-6):
                continue
            projected,_ = cv2.projectPoints(world,rvec,tvec,k,distortion)
            errors = np.linalg.norm(projected.reshape(-1,2)-pixels,axis=1)
            choices.append((float(np.mean(errors**2)),ground_from_camera,errors))
    if not choices:
        raise ValueError("no valid camera pose above the ground with control points in front")
    _,transform,fit_errors = min(choices,key=lambda item:item[0])
    camera = {'intrinsics':k.tolist(),'distortion':distortion.tolist(),
              'image_width':data['image_width'],'image_height':data['image_height'],
              'world_from_camera':transform.tolist()}
    checks = evaluate_camera(scene,camera,data['check_points'])
    pixel_limit = float(data.get('max_reprojection_error_px',3))
    position_limit = float(data.get('max_position_error_m',.2))
    if not np.isfinite([pixel_limit,position_limit]).all() or min(pixel_limit,position_limit) <= 0:
        raise ValueError("validation limits must be finite and positive")
    passed = (float(fit_errors.max()) <= pixel_limit and checks['reprojection_max_px'] <= pixel_limit
              and checks['position_max_m'] <= position_limit)
    report = {'passed':bool(passed),'fit_points':len(world),'fit_errors_px':fit_errors.tolist(),
              'fit_rmse_px':float(np.sqrt(np.mean(fit_errors**2))),'fit_max_px':float(fit_errors.max()),
              'independent_ground_check':checks,'max_reprojection_error_px':pixel_limit,
              'max_position_error_m':position_limit,'scene_sha256':scene.sha256,
              'geometry_sha256':scene.geometry_sha256,
              'example_only':bool(scene.example_only or data.get('example_only',True)),
              'height_source':'registered_ground_plane; not monocular depth estimation'}
    return camera,report
