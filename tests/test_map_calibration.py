import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from yudun.geometry import transform_points
from yudun.map3d.scene import MapScene
from yudun.map3d.calibration import calibrate_camera, evaluate_camera


def fixture(tmp_path,nonplanar=False):
    cv2 = pytest.importorskip('cv2')
    np.save(tmp_path/'cloud.npy',[[0,0,0],[1,0,0],[0,1,0]])
    mapping = np.eye(4)
    mapping[:3,:3] = Rotation.from_euler('zy',[.3,.15]).as_matrix()
    mapping[:3,3] = [12,-7,2]
    path = tmp_path/'map.json'
    path.write_text(json.dumps({'schema_version':1,'frame_id':'warehouse_3d','ground_frame_id':'map',
                               'geometry':'cloud.npy','unit_to_m':1,'example_only':True,
                               'map_from_ground':mapping.tolist()}))
    scene = MapScene(path)
    pose = np.array([[1,0,0,0],[0,-.6,.8,-8],[0,-.8,-.6,6],[0,0,0,1]],float)
    source = np.array([[-3,-2,0],[3,-2,0],[3,3,0],[-3,3,0],[0,-1,0],[1,2,0]],float)
    if nonplanar:
        source[-1,2] = 1
    check = np.array([[-2.5,.5,0],[2,.25,0],[0,4,0],[1,-.2,0]],float)
    camera_from_map = np.linalg.inv(mapping@pose)
    k = np.array([[800,0,640],[0,810,360],[0,0,1]],float)
    distortion = [.02,-.01,.001,-.001,0]
    rvec,_ = cv2.Rodrigues(camera_from_map[:3,:3])
    def pairs(ground):
        world = transform_points(ground,mapping)
        pixels,_ = cv2.projectPoints(world,rvec,camera_from_map[:3,3],k,np.array(distortion))
        return {'map_points_m':world.tolist(),'image_points_px':pixels.reshape(-1,2).tolist()}
    data = {'intrinsics':k.tolist(),'distortion':distortion,'image_width':1280,'image_height':720,
            'control_points':pairs(source),'check_points':pairs(check),'example_only':False}
    return scene,pose,data


@pytest.mark.parametrize('nonplanar',[False,True])
def test_camera_calibration_roundtrip_tilted_map_and_independent_checks(tmp_path,nonplanar):
    scene,expected,data = fixture(tmp_path,nonplanar)
    camera,report = calibrate_camera(scene,data)
    np.testing.assert_allclose(camera['world_from_camera'],expected,atol=1e-6)
    assert report['passed'] and report['example_only']
    assert report['independent_ground_check']['position_max_m'] < 1e-6
    assert report['fit_max_px'] < 1e-6


def test_bad_independent_points_fail_accuracy_gate(tmp_path):
    scene,_,data = fixture(tmp_path)
    data['check_points']['image_points_px'][0][0] += 80
    _,report = calibrate_camera(scene,data)
    assert not report['passed']
    assert report['independent_ground_check']['position_max_m'] > .2


def test_reused_fit_points_and_non_ground_checkpoints_rejected(tmp_path):
    scene,_,data = fixture(tmp_path)
    data['check_points'] = {key:values[:3] for key,values in data['control_points'].items()}
    with pytest.raises(ValueError,match='independent'):
        calibrate_camera(scene,data)
    scene,pose,data = fixture(tmp_path)
    camera = {key:data[key] for key in ('intrinsics','distortion','image_width','image_height')}
    camera['world_from_camera'] = pose.tolist()
    data['check_points']['map_points_m'][0][2] += 1
    with pytest.raises(ValueError,match='ground plane'):
        evaluate_camera(scene,camera,data['check_points'])
