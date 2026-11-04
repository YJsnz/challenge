import json
import struct
import numpy as np
import pytest
from yudun.geometry import transform_points
from yudun.map3d import MapScene, fit_registration
from yudun.map3d.cloud import load_cloud, _lzf


@pytest.fixture
def map_scene(tmp_path):
    np.save(tmp_path/'cloud.npy',[[0,0,0],[1000,0,0],[0,1000,2000]])
    transform = [[0,-1,0,10],[0,0,-1,20],[1,0,0,30],[0,0,0,1]]
    data = {'schema_version':1,'frame_id':'warehouse_3d','ground_frame_id':'map',
            'example_only':True,'geometry':'cloud.npy','unit_to_m':.001,
            'map_from_ground':transform}
    path = tmp_path/'scene.json'
    path.write_text(json.dumps(data))
    return MapScene(path)


def test_metric_map_units_and_ground_registration(map_scene):
    np.testing.assert_allclose(map_scene.cloud,[[0,0,0],[1,0,0],[0,1,2]])
    np.testing.assert_allclose(map_scene.lift([[2,3]]),[[7,20,32]])
    np.testing.assert_allclose(map_scene.to_ground([[7,20,32]]),[[2,3,0]])
    xyz = map_scene.lift([[0,0],[3,4]])
    assert np.linalg.norm(xyz[1]-xyz[0]) == pytest.approx(5)


def test_registration_fits_rotation_translation_without_scale():
    source = np.array([[0,0,0],[2,0,0],[0,3,0],[2,3,0]],float)
    target = np.array([[10,20,30],[10,20,32],[7,20,30],[7,20,32]],float)
    transform, report = fit_registration(source,target)
    np.testing.assert_allclose(transform_points(source,transform),target,atol=1e-12)
    assert report['fit_rmse_m'] < 1e-12 and report['scale_estimated'] is False
    with pytest.raises(ValueError,match='collinear'):
        fit_registration([[0,0,0],[1,0,0],[2,0,0]],[[0,0,0],[1,0,0],[2,0,0]])


def test_lift_tracks_velocity_covariance_and_predictions(map_scene):
    track = {'id':1,'category':'person','status':'POSSIBLY_OCCLUDED','state':[2,3,4,5],
             'covariance':np.diag([.25,1,2,2]).tolist(),
             'predictions':[{'horizon':1,'mean':[6,8],'covariance':[[1,0],[0,4]]}]}
    frame = {'frame_id':'map','tracks':[track]}
    result = map_scene.project_result(frame)
    mapped = result['map_3d']['tracks'][0]
    np.testing.assert_allclose(mapped['position_m'],[7,20,32])
    np.testing.assert_allclose(mapped['velocity_mps'],[-5,0,4])
    np.testing.assert_allclose(mapped['covariance_m2'],np.diag([1,0,.25]))
    np.testing.assert_allclose(mapped['predictions'][0]['position_m'],[2,20,36])
    assert mapped['status'] == 'POSSIBLY_OCCLUDED'
    assert 'map_3d' not in frame and frame['tracks'][0]['state'] == [2,3,4,5]
    result['map_3d']['scene_sha256'] = 'wrong-map'
    with pytest.raises(ValueError,match='different map'):
        map_scene.project_result(result)


def test_map_rejects_bad_registration_and_changed_geometry(map_scene):
    data = dict(map_scene.data)
    data['map_from_ground'][0][0] = 2
    map_scene.path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='rotation'):
        MapScene(map_scene.path)
    data['map_from_ground'] = None
    data['geometry_sha256'] = 'incorrect'
    map_scene.path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash'):
        MapScene(map_scene.path)
    data.pop('geometry_sha256')
    map_scene.path.write_text(json.dumps(data))
    scene = MapScene(map_scene.path)
    assert not scene.registered
    with pytest.raises(ValueError,match='registration'):
        scene.lift([[0,0]])


@pytest.mark.parametrize('mode',['ascii','binary','binary_compressed'])
def test_pcd_formats_xyz_and_rgb(tmp_path,mode):
    header = f'VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA {mode}\n'.encode()
    if mode == 'ascii':
        data = b'1 2 3 16711680\n4 5 6 65280\n'
    elif mode == 'binary':
        data = struct.pack('<fffIfffI',1,2,3,0xff0000,4,5,6,0x00ff00)
    else:
        raw = struct.pack('<ffffffII',1,4,2,5,3,6,0xff0000,0x00ff00)
        packed = bytes([len(raw)-1])+raw
        data = struct.pack('<II',len(packed),len(raw))+packed
    path = tmp_path/'map.pcd'
    path.write_bytes(header+data)
    points,colours,total = load_cloud(path)
    np.testing.assert_allclose(points,[[1,2,3],[4,5,6]])
    np.testing.assert_array_equal(colours,[[255,0,0],[0,255,0]])
    assert total == 2


@pytest.mark.parametrize('mode',['ascii','binary_little_endian','binary_big_endian'])
def test_ply_formats_with_faces_after_vertices(tmp_path,mode):
    header = f'ply\nformat {mode} 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'.encode()
    data = b'1 2 3 255 0 0\n4 5 6 0 255 0\n' if mode == 'ascii' else struct.pack(('<' if 'little' in mode else '>')+'fffBBBfffBBB',1,2,3,255,0,0,4,5,6,0,255,0)
    path = tmp_path/'map.ply'
    path.write_bytes(header+data)
    points,colours,total = load_cloud(path,max_points=1)
    np.testing.assert_allclose(points,[[1,2,3]])
    np.testing.assert_array_equal(colours,[[255,0,0]])
    assert total == 2


def test_cloud_filters_nan_and_rejects_truncated_compression(tmp_path):
    path = tmp_path/'map.npy'
    np.save(path,[[0,1,2],[float('nan'),0,0],[3,4,5]])
    points,_,total = load_cloud(path)
    assert len(points) == 2 and total == 3
    assert _lzf(b'\x00a\x20\x00',4) == b'aaaa'
    with pytest.raises(ValueError):
        _lzf(b'\x04a',5)
    with pytest.raises(ValueError):
        _lzf(b'\x20\x05',3)
