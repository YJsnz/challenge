"""Video-inspired demonstration geometry; never a surveyed warehouse map."""
from dataclasses import asdict
from importlib.resources import files
import json
from pathlib import Path
import shutil
import numpy as np
from ..adapters import camera_observation
from ..geometry import transform_points
from ..forklift_operator import PROFILE
from ..pipeline import load_config, Pipeline
from .scene import MapScene, digest, fit_registration
from .workflows import write


def _look_at(position,target):
    position,target = np.asarray(position,float),np.asarray(target,float)
    forward = target-position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward,[0,0,1.])
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3,:3] = np.column_stack([right,np.cross(forward,right),forward])
    pose[:3,3] = position
    return pose


def _cloud(layout):
    blocks = []
    def surface(bounds,colour,step=.14):
        low,high = np.array(bounds[:3]),np.array(bounds[3:])
        for fixed in range(3):
            axes = [axis for axis in range(3) if axis != fixed]
            u,v = [np.linspace(low[a],high[a],max(2,int((high[a]-low[a])/step)+1)) for a in axes]
            uu,vv = np.meshgrid(u,v)
            for value in (low[fixed],high[fixed]):
                xyz = np.empty((uu.size,3))
                xyz[:,axes[0]],xyz[:,axes[1]],xyz[:,fixed] = uu.ravel(),vv.ravel(),value
                blocks.append(np.column_stack([xyz,np.tile(colour,(len(xyz),1))]))
    x0,y0,x1,y1 = layout['ground_bounds_m']
    surface([x0,y0,-.03,x1,y1,0],[185,197,202],.2)
    for x0,y0,x1,y1 in layout['racks_xy_m']:
        height = layout['rack_height_m']
        for z in (1.2,2.5,3.8):
            surface([x0,y0,z,x1,y1,z+.08],[42,99,105],.2)
        for x in (x0,x1):
            for y in np.linspace(y0,y1,max(2,int((y1-y0)/2.5)+1)):
                surface([x-.03,y-.03,0,x+.03,y+.03,height],[42,99,105],.25)
        for y in np.arange(y0+.15,y1-.5,2):
            for z in (.15,1.4,2.7):
                surface([x0+.1,y,z,x1-.1,min(y+1.5,y1-.1),z+.8],[189,153,105],.24)
    for x0,y0,x1,y1 in layout['pallets_xy_m']:
        surface([x0,y0,0,x1,y1,1.1],[193,157,111],.14)
    return np.concatenate(blocks)


def make_demo(output,source_calibration=None,model=None):
    import cv2
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    asset = files('yudun').joinpath('assets')
    layout = json.loads(asset.joinpath('warehouse026-demo-layout.json').read_text(encoding='utf-8'))
    shutil.copyfile(str(asset.joinpath('warehouse026-demo-plan.png')),output/'reference-plan.png')
    write(output/'layout-assumptions.json',layout)
    a,b = np.deg2rad(20),np.deg2rad(8)
    rz = np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    rx = np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
    mapping = np.eye(4)
    mapping[:3,:3],mapping[:3,3] = rz@rx,[12,-4,1]
    cloud = _cloud(layout)
    cloud[:,:3] = transform_points(cloud[:,:3],mapping)
    np.save(output/'cloud.npy',cloud)
    controls = np.array([[-3,-6,0],[3,-6,0],[3,7,0],[-3,7,0]],float)
    transform,registration = fit_registration(controls,transform_points(controls,mapping))
    registration.update({'example_only':True,'provenance':'synthetic_exact_correspondences_not_surveyed'})
    write(output/'registration-report.json',registration)
    spec = {'schema_version':1,'frame_id':'warehouse_demo_3d','ground_frame_id':'map','example_only':True,
            'geometry':'cloud.npy','geometry_sha256':digest(output/'cloud.npy'),'unit_to_m':1,
            'map_from_ground':transform.tolist(),'ground_bounds_m':layout['ground_bounds_m'],
            'ground_obstacles_m':layout['racks_xy_m']+layout['pallets_xy_m'],
            'reference_image':'reference-plan.png','provenance':layout['provenance'],
            'dimension_status':layout['dimension_status']}
    write(output/'map.json',spec)
    scene = MapScene(output/'map.json',load_points=False)
    pose = _look_at([0,-12,4],[0,0,0])
    k = np.array([[800,0,640],[0,800,360],[0,0,1]],float)
    width,height = 1280,720
    provenance = 'assumed_demo_camera_pose'
    if source_calibration is not None:
        source_calibration = Path(source_calibration).resolve()
        source = json.loads(source_calibration.read_text(encoding='utf-8-sig'))
        sensor = next(s for s in source['sensors'] if s['id'] == 'Camera_0000')
        extrinsic = np.eye(4)
        extrinsic[:3] = np.asarray(sensor['extrinsicMatrix'],float)
        publisher_pose = np.linalg.inv(extrinsic)
        forward = publisher_pose[:2,2]/np.linalg.norm(publisher_pose[:2,2])
        right = np.array([forward[1],-forward[0]])
        ground_from_publisher = np.eye(4)
        ground_from_publisher[:2,:2] = np.vstack([right,forward])
        ground_from_publisher[:2,3] = [0,-10]-ground_from_publisher[:2,:2]@publisher_pose[:2,3]
        pose = ground_from_publisher@publisher_pose
        k = np.asarray(sensor['intrinsicMatrix'],float)
        attrs = {v['name']:v['value'] for v in sensor['attributes']}
        width,height = int(attrs['frameWidth']),int(attrs['frameHeight'])
        forward_h = k@extrinsic[:3,[0,1,3]]
        inverse_h = np.linalg.inv(forward_h)
        stored = np.asarray(sensor['homography'],float)
        error = float(np.max(np.abs(inverse_h/inverse_h[2,2]-stored/stored[2,2])))
        if error > 1e-6:
            raise ValueError("publisher homography and camera extrinsic convention disagree")
        provenance = 'publisher_intrinsics_extrinsics_on_illustrative_map_not_field_validated'
        write(output/'publisher-camera-check.json',{'source_calibration_sha256':digest(source_calibration),
               'homography_inverse_consistency_error':error,'ground_from_publisher':ground_from_publisher.tolist(),
               'example_only':True,'metric_scale_status':'publisher numeric scale adopted for demo; not independently surveyed'})
    config = load_config()
    config.update({'bounds':layout['ground_bounds_m'],'obstacles':spec['ground_obstacles_m'],
                   'sensor_positions':{'camera':pose[:2,3].tolist()},'map_scene':'map.json'})
    if source_calibration is not None and digest(source_calibration) == layout["source_calibration_sha256"]:
        config["forklift_operator_profile"] = PROFILE
    write(output/'tracking_scene.json',config)
    camera = {'frame_id':'camera_optical_frame','intrinsics':k.tolist(),'distortion':[],
              'world_from_camera':pose.tolist(),'map_from_camera':(mapping@pose).tolist(),
              'image_width':width,'image_height':height,'input_size':640,
              'model':str(Path(model).resolve()) if model else 'REPLACE_WITH_INDUSTRIAL_MODEL.onnx',
              'class_names':['forklift_or_cart','person'],'measurement_covariance':[[.25,0],[0,.25]],
              'detector_options':{'class_confidence':{'person':.15,'forklift_or_cart':.4}},
              'map_scene':'map.json','map_scene_sha256':scene.sha256,
              'scene_config':'tracking_scene.json','scene_config_sha256':digest(output/'tracking_scene.json'),
              'provenance':provenance}
    if model:
        camera['model_sha256'] = digest(model)
    write(output/'camera.example.json',{'schema_version':1,'frame_id':'map','example_only':True,'camera':camera})
    # Demonstrate a real calibration workflow on simulated, explicitly labelled pairs.
    camera_from_ground = np.linalg.inv(pose)
    rvec,_ = cv2.Rodrigues(camera_from_ground[:3,:3])
    def pairs(ground):
        ground = np.asarray(ground,float)
        pixels,_ = cv2.projectPoints(ground,rvec,camera_from_ground[:3,3],k,None)
        return {'map_points_m':scene.lift(ground[:,:2]).tolist(),'image_points_px':pixels.reshape(-1,2).tolist()}
    request = {'schema_version':1,'example_only':True,'map_scene':'map.json',
               **{key:camera[key] for key in ('intrinsics','distortion','image_width','image_height','model','class_names')},
               'control_points':pairs([[-1,-4,0],[1,-4,0],[1,5,0],[-1,5,0],[0,-1,0],[.6,3,0]]),
               'check_points':pairs([[-.8,-2,0],[.8,1,0],[-.4,4,0],[.4,-3,0]]),
               'measurement_covariance_m2':[[.25,0],[0,.25]],'max_position_error_m':.01,
               'max_reprojection_error_px':.1,'provenance':'synthetic_pairs_from_demo_pose_not_measured_controls'}
    write(output/'calibration-request.example.json',request)
    inputs,errors = [],[]
    for index in range(61):
        t = index/10
        observations,truth,boxes = [],[],[]
        for identity,category,xy in ((1,'person',[-1.2+.3*t,-4]),(2,'forklift_or_cart',[.4,-8+1.8*t])):
            pixel,_ = cv2.projectPoints(np.array([[*xy,0]],float),rvec,camera_from_ground[:3,3],k,None)
            u,v = pixel.reshape(2)
            if not (0 <= u < width and 0 <= v < height):
                continue
            box = [float(u-12),float(v-60),float(u+12),float(v)]
            obs = camera_observation(box,category,.95,t,k,pose,[[.02,0],[0,.02]],observation_id=f'demo:{identity}:{index}')
            observations.append(asdict(obs))
            boxes.append({'bbox':box,'category':category,'confidence':.95})
            truth.append({'id':identity,'category':category,'position':xy,'radius':.35 if identity == 1 else .6})
            errors.append(float(np.linalg.norm(scene.lift([obs.position])[0]-scene.lift([xy])[0])))
        inputs.append({'schema_version':1,'frame_id':'map','timestamp':t,'sensors':['camera'],
                       'observations':observations,'ground_truth':truth,'visual_detections':boxes,
                       'source_frame':index,'provenance':'synthetic_geometry_not_detector_output'})
    with (output/'observations.jsonl').open('x',encoding='utf-8') as stream:
        for frame in inputs:
            stream.write(json.dumps(frame)+'\n')
    pipeline = Pipeline(load_config(output/'tracking_scene.json'))
    with (output/'results.jsonl').open('x',encoding='utf-8') as stream:
        for frame in inputs:
            stream.write(json.dumps(pipeline.process(frame))+'\n')
    write(output/'demo-validation.json',{'example_only':True,'frames':len(inputs),
          'pixel_ground_map_max_numerical_error_m':max(errors,default=0),
          'note':'Exact simulated geometry only; not real-video location accuracy or detector performance',
          'map_scene_sha256':scene.sha256,'source_reference_video_sha256':layout['source_video_sha256']})
    return {'map_scene':str(scene.path),'scene_config':str(output/'tracking_scene.json'),
            'calibration':str(output/'camera.example.json'),'input':str(output/'observations.jsonl'),
            'results':str(output/'results.jsonl'),'reference_image':str(output/'reference-plan.png')}
