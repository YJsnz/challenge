"""File-based map registration and camera calibration workflows shared by CLI/Qt."""
import json
from pathlib import Path
import numpy as np
from ..geometry import transform_points
from ..pipeline import load_config
from ..schema import CLASSES, covariance
from .scene import MapScene, digest, fit_registration, points
from .calibration import calibrate_camera, evaluate_camera, independent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def register_map(request_path,output):
    request_path,output = Path(request_path).resolve(),Path(output).resolve()
    data = read(request_path)
    geometry = (request_path.parent/data['geometry']).resolve()
    transform,report = fit_registration(data['ground_points_m'],data['map_points_m'])
    limit = float(data.get('max_registration_error_m',.05))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("registration error limit must be finite and positive")
    example = data.get('example_only',True)
    if not isinstance(example,bool):
        raise ValueError("example_only must be boolean")
    check = data.get('check_points')
    if not check and not example:
        raise ValueError("measured map registration requires independent checkpoints")
    passed = report['fit_max_m'] <= limit
    if check:
        source,target = points(check['ground_points_m'],minimum=3),points(check['map_points_m'],minimum=3)
        if source.shape != target.shape or np.max(np.abs(source[:,2])) > 1e-6:
            raise ValueError("registration checkpoints must be matching ground-plane pairs")
        independent(points(data['ground_points_m']),source)
        errors = np.linalg.norm(transform_points(source,transform)-target,axis=1)
        report['independent_check'] = {'errors_m':errors.tolist(),'rmse_m':float(np.sqrt(np.mean(errors**2))),
                                       'maximum_m':float(errors.max())}
        passed = passed and errors.max() <= limit
    report.update({'passed':bool(passed),'example_only':example,'max_registration_error_m':limit,
                   'request_sha256':digest(request_path),'geometry_sha256':digest(geometry)})
    output.mkdir(parents=True,exist_ok=False)
    write(output/'registration-report.json',report)
    if not passed:
        raise ValueError(f"registration failed error limits; inspect {output/'registration-report.json'}")
    spec = {'schema_version':1,'frame_id':data.get('frame_id','warehouse_3d'),'ground_frame_id':'map',
            'geometry':str(geometry),'geometry_sha256':digest(geometry),'unit_to_m':data['unit_to_m'],
            'map_from_geometry':data.get('map_from_geometry',np.eye(4).tolist()),
            'map_from_ground':transform.tolist(),'example_only':example,
            'registration_report':'registration-report.json'}
    for field in ('ground_bounds_m','ground_obstacles_m','reference_image','provenance'):
        if field in data:
            spec[field] = str((request_path.parent/data[field]).resolve()) if field == 'reference_image' else data[field]
    write(output/'map.json',spec)
    MapScene(output/'map.json',load_points=False)
    return output/'map.json'


def calibrate(request_path,output):
    request_path,output = Path(request_path).resolve(),Path(output).resolve()
    data = read(request_path)
    scene = MapScene(request_path.parent/data['map_scene'],load_points=False)
    camera,report = calibrate_camera(scene,data)
    model = (request_path.parent/data['model']).resolve()
    names = data['class_names']
    if not model.is_file() or not names or len(set(names)) != len(names) or not set(names) <= CLASSES:
        raise ValueError("supply an existing industrial model and its explicit mapped class names")
    bounds = np.asarray(data.get('ground_bounds_m',scene.data.get('ground_bounds_m')),float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[:2] >= bounds[2:]):
        raise ValueError("explicit ground_bounds_m required for the tracking scene")
    measurement = data.get('measurement_covariance_m2')
    if measurement is None and not report['example_only']:
        raise ValueError("measured calibration requires measurement_covariance_m2")
    measurement = covariance(measurement if measurement is not None else [[.25,0],[0,.25]],2)
    output.mkdir(parents=True,exist_ok=False)
    report['request_sha256'] = digest(request_path)
    report['model_sha256'] = digest(model)
    write(output/'camera-report.json',report)
    if not report['passed']:
        raise ValueError(f"camera validation failed; inspect {output/'camera-report.json'}")
    config = load_config((request_path.parent/data['scene_template']).resolve() if data.get('scene_template') else None)
    config.update({'map_scene':str(scene.path),'bounds':bounds.tolist(),
                   'obstacles':scene.data.get('ground_obstacles_m',[]),
                   'sensor_positions':{'camera':np.asarray(camera['world_from_camera'])[:2,3].tolist()}})
    write(output/'tracking_scene.json',config)
    camera.update({'frame_id':'camera_optical_frame','model':str(model),'model_sha256':digest(model),
                   'class_names':names,'input_size':data.get('input_size',640),
                   'measurement_covariance':measurement.tolist(),'detector_options':data.get('detector_options',{}),
                   'map_scene':str(scene.path),'map_scene_sha256':scene.sha256,
                   'scene_config':'tracking_scene.json','scene_config_sha256':digest(output/'tracking_scene.json'),
                   'map_from_camera':(scene.map_from_ground@np.array(camera['world_from_camera'])).tolist(),
                   'control_map_points_m':data['control_points']['map_points_m'],
                   'validation_report':'camera-report.json','validation_report_sha256':digest(output/'camera-report.json')})
    calibration = {'schema_version':1,'frame_id':'map','example_only':report['example_only'],'camera':camera}
    write(output/'camera.json',calibration)
    return output/'camera.json'


def check_camera(request_path):
    request_path = Path(request_path).resolve()
    data = read(request_path)
    calibration_path = (request_path.parent/data['calibration']).resolve()
    calibration = read(calibration_path)
    camera = calibration['camera']
    scene = MapScene(calibration_path.parent/camera['map_scene'],load_points=False)
    if camera.get('map_scene_sha256') and camera['map_scene_sha256'] != scene.sha256:
        raise ValueError("calibration references a different map scene hash")
    if not camera.get('control_map_points_m'):
        raise ValueError("camera has no fitting controls to establish checkpoint independence")
    independent(points(camera['control_map_points_m']),points(data['check_points']['map_points_m'],minimum=3))
    report = evaluate_camera(scene,camera,data['check_points'])
    report.update({'calibration_sha256':digest(calibration_path),'scene_sha256':scene.sha256,
                   'example_only':bool(calibration.get('example_only',True) or scene.example_only),
                   'height_source':'registered_ground_plane'})
    return report
