import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from ..geometry import transform_points
from .cloud import load_cloud


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def rigid(value):
    matrix = np.asarray(value,float)
    transform_points(np.empty((0,3)),matrix)
    return matrix


def points(value, dimensions=3, minimum=1):
    value = np.asarray(value,float)
    if value.ndim != 2 or value.shape[1] != dimensions or len(value) < minimum or not np.isfinite(value).all():
        raise ValueError(f"expected at least {minimum} finite {dimensions}-D points")
    return value


def fit_registration(ground_points_m, map_points_m):
    source,target = points(ground_points_m,minimum=3),points(map_points_m,minimum=3)
    if source.shape != target.shape or np.max(np.abs(source[:,2])) > 1e-6:
        raise ValueError("registration pairs must have matching shape and ground z=0")
    a,b = source.mean(axis=0),target.mean(axis=0)
    if np.linalg.matrix_rank(source-a,tol=1e-8) < 2 or np.linalg.matrix_rank(target-b,tol=1e-8) < 2:
        raise ValueError("registration points must not be collinear")
    u,_,vt = np.linalg.svd((source-a).T@(target-b))
    correction = np.eye(3)
    correction[2,2] = np.linalg.det(vt.T@u.T)
    rotation = vt.T@correction@u.T
    transform = np.eye(4)
    transform[:3,:3],transform[:3,3] = rotation,b-rotation@a
    residuals = np.linalg.norm(transform_points(source,transform)-target,axis=1)
    return transform,{'residuals_m':residuals.tolist(),'fit_rmse_m':float(np.sqrt(np.mean(residuals**2))),
                      'fit_max_m':float(residuals.max()),'scale_estimated':False}


class MapScene:
    def __init__(self,path,load_points=True,max_points=50000):
        self.path = Path(path).resolve()
        self.data = json.loads(self.path.read_text(encoding='utf-8-sig'))
        d = self.data
        if d.get('schema_version') != 1 or d.get('ground_frame_id') != 'map':
            raise ValueError("map scene requires schema 1 and ground_frame_id=map")
        if not isinstance(d.get('frame_id'),str) or not d['frame_id'].strip() or d['frame_id'] == 'map':
            raise ValueError("3-D map frame must have an explicit name distinct from the ground frame 'map'")
        self.frame_id,self.ground_frame_id = d['frame_id'],d['ground_frame_id']
        if not isinstance(d.get('example_only',True),bool):
            raise ValueError("example_only must be boolean")
        self.example_only = d.get('example_only',True)
        self.unit_to_m = float(d['unit_to_m'])
        if not np.isfinite(self.unit_to_m) or self.unit_to_m <= 0:
            raise ValueError("map unit_to_m must be finite and positive")
        self.geometry_path = (self.path.parent/d['geometry']).resolve()
        self.geometry_sha256 = digest(self.geometry_path)
        if d.get('geometry_sha256') and d['geometry_sha256'] != self.geometry_sha256:
            raise ValueError("map geometry hash mismatch")
        self.sha256 = digest(self.path)
        self.map_from_geometry = rigid(d.get('map_from_geometry',np.eye(4)))
        self.map_from_ground = rigid(d['map_from_ground']) if d.get('map_from_ground') is not None else None
        self.cloud,self.colours,self.original_points = None,None,0
        if load_points:
            cloud,self.colours,self.original_points = load_cloud(self.geometry_path,max_points)
            self.cloud = transform_points(cloud*self.unit_to_m,self.map_from_geometry)

    @property
    def registered(self):
        return self.map_from_ground is not None

    def lift(self,xy):
        if not self.registered:
            raise ValueError("map has no ground-to-map registration")
        xy = points(xy,2)
        return transform_points(np.column_stack([xy,np.zeros(len(xy))]),self.map_from_ground)

    def to_ground(self,xyz):
        if not self.registered:
            raise ValueError("map has no ground-to-map registration")
        return transform_points(points(xyz),np.linalg.inv(self.map_from_ground))

    def project_result(self,frame):
        if frame.get('frame_id') != self.ground_frame_id:
            raise ValueError("tracking frame differs from map ground frame")
        if not self.registered:
            raise ValueError("load a registered map before projecting tracks")
        existing = frame.get('map_3d')
        if existing and existing.get('scene_sha256') != self.sha256:
            raise ValueError("result was registered against a different map scene")
        if frame.get('map_scene_sha256') and frame['map_scene_sha256'] != self.sha256:
            raise ValueError("result belongs to a different map scene")
        basis = self.map_from_ground[:3,:2]
        def location(xy,covariance):
            covariance = np.asarray(covariance,float)
            if covariance.shape != (2,2) or not np.isfinite(covariance).all():
                raise ValueError("ground position covariance must be finite 2x2")
            return {'position_m':self.lift([xy])[0].tolist(),
                    'covariance_m2':(basis@covariance@basis.T).tolist()}
        tracks = []
        for track in frame.get('tracks',[]):
            state = np.asarray(track['state'],float)
            if state.shape != (4,) or not np.isfinite(state).all():
                raise ValueError("expected finite ground state [x,y,vx,vy]")
            track3d = {'id':track['id'],'category':track['category'],'status':track['status'],
                       'velocity_mps':(basis@state[2:]).tolist(),
                       **location(state[:2],np.asarray(track['covariance'])[:2,:2]),
                       'predictions':[{**p,**location(p['mean'],p['covariance'])} for p in track.get('predictions',[])]}
            tracks.append(track3d)
        result = copy.copy(frame)
        result['map_3d'] = {'frame_id':self.frame_id,'ground_frame_id':self.ground_frame_id,
                            'scene_sha256':self.sha256,'geometry_sha256':self.geometry_sha256,
                            'example_only':self.example_only or bool(frame.get('example_calibration')),
                            'height_source':'registered_ground_plane',
                            'covariance_scope':'ground_position_only_excludes_registration_uncertainty',
                            'tracks':tracks,
                            'observations':[{'observation_id':o.get('observation_id',''),
                                'category':o['category'],**location(o['position'],o['covariance'])}
                                for o in frame.get('observations',[])]}
        return result
