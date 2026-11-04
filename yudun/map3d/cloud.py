"""PCD and PLY point-cloud input; numpy files never enable pickle loading."""
from pathlib import Path
import struct
import numpy as np


def _lzf(data, expected):
    out, index = bytearray(), 0
    while index < len(data):
        token = data[index]
        index += 1
        if token < 32:
            length = token+1
            if index+length > len(data):
                raise ValueError("truncated LZF literal")
            out.extend(data[index:index+length])
            index += length
        else:
            length, offset = token >> 5, (token & 31) << 8
            if length == 7:
                if index >= len(data):
                    raise ValueError("truncated LZF length")
                length += data[index]
                index += 1
            if index >= len(data):
                raise ValueError("truncated LZF reference")
            offset += data[index]+1
            index += 1
            start = len(out)-offset
            if start < 0:
                raise ValueError("invalid LZF reference")
            if len(out)+length+2 > expected:
                raise ValueError("LZF output exceeds declared size")
            for j in range(length+2):
                out.append(out[start+j])
        if len(out) > expected:
            raise ValueError("LZF output exceeds declared size")
    if len(out) != expected:
        raise ValueError("LZF output size mismatch")
    return bytes(out)


def _pcd(stream):
    header = {}
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError("PCD DATA header missing")
        line = raw.decode('ascii').strip()
        if not line or line.startswith('#'):
            continue
        key, *value = line.split()
        header[key.upper()] = value
        if key.upper() == 'DATA':
            break
    names = header['FIELDS']
    sizes = list(map(int, header['SIZE']))
    kinds = header['TYPE']
    counts = list(map(int, header.get('COUNT', ['1']*len(names))))
    number = int(header.get('POINTS', [str(int(header['WIDTH'][0])*int(header.get('HEIGHT', ['1'])[0]))])[0])
    if number <= 0 or not len(names) == len(sizes) == len(kinds) == len(counts) or min(counts) <= 0:
        raise ValueError("invalid PCD dimensions")
    if len(set(names)) != len(names) or not {'x','y','z'} <= set(names):
        raise ValueError("PCD requires unique x/y/z fields")
    if any(counts[names.index(axis)] != 1 for axis in ('x','y','z')):
        raise ValueError("PCD x/y/z must be scalar fields")
    codes = {'F':'f', 'I':'i', 'U':'u'}
    types = [np.dtype('<'+codes[k]+str(size)) for k,size in zip(kinds,sizes)]
    fields = [(name, dtype, (count,)) if count > 1 else (name, dtype)
              for name,dtype,count in zip(names,types,counts)]
    dtype = np.dtype(fields)
    mode = header['DATA'][0]
    if mode == 'ascii':
        values = np.loadtxt(stream, max_rows=number, ndmin=2)
        if values.shape != (number,sum(counts)):
            raise ValueError("PCD point count or field count differs from header")
        offsets = np.cumsum([0]+counts[:-1])
        columns = {name: values[:,offset] for name,offset in zip(names,offsets)}
    elif mode == 'binary':
        data = stream.read(number*dtype.itemsize)
        if len(data) != number*dtype.itemsize:
            raise ValueError("truncated binary PCD")
        values = np.frombuffer(data, dtype=dtype, count=number)
        columns = {name: values[name] for name in names}
    elif mode == 'binary_compressed':
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated compressed PCD header")
        compressed, expanded = struct.unpack('<II', prefix)
        if expanded != number*dtype.itemsize:
            raise ValueError("compressed PCD field size mismatch")
        packed = stream.read(compressed)
        if len(packed) != compressed:
            raise ValueError("truncated compressed PCD data")
        data = _lzf(packed, expanded)
        columns, offset = {}, 0
        for name,typ,count in zip(names,types,counts):
            values = np.frombuffer(data, dtype=typ, count=number*count, offset=offset).reshape(number,count)
            columns[name] = values[:,0] if count == 1 else values
            offset += number*typ.itemsize*count
    else:
        raise ValueError("unsupported PCD DATA encoding")
    xyz = np.column_stack([columns[axis] for axis in ('x','y','z')])
    colours = None
    if 'rgb' in columns or 'rgba' in columns:
        name = 'rgb' if 'rgb' in columns else 'rgba'
        packed = columns[name]
        if kinds[names.index(name)] == 'F':
            packed = np.asarray(packed,dtype='<f4').view('<u4')
        else:
            packed = np.asarray(packed,dtype='<u4')
        colours = np.column_stack([(packed >> shift) & 255 for shift in (16,8,0)]).astype(np.uint8)
    return xyz, colours


def _ply(stream):
    if stream.readline().strip() != b'ply':
        raise ValueError("invalid PLY signature")
    props, number, mode, active, first = [], None, None, None, None
    codes = {'char':'i1','uchar':'u1','short':'i2','ushort':'u2','int':'i4','uint':'u4',
             'float':'f4','double':'f8','int8':'i1','uint8':'u1','int16':'i2','uint16':'u2',
             'int32':'i4','uint32':'u4','float32':'f4','float64':'f8'}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PLY end_header missing")
        parts = line.decode('ascii').split()
        if not parts:
            continue
        if parts[0] == 'end_header':
            break
        if parts[0] == 'format':
            mode = parts[1]
        elif parts[0] == 'element':
            active = parts[1]
            first = first or active
            if active == 'vertex':
                number = int(parts[2])
        elif parts[0] == 'property' and active == 'vertex':
            if parts[1] == 'list':
                raise ValueError("list-valued PLY vertex properties are unsupported")
            props.append((parts[2],codes[parts[1]]))
    names = [name for name,_ in props]
    if first != 'vertex' or not number or number < 0 or not {'x','y','z'} <= set(names) or len(set(names)) != len(names):
        raise ValueError("PLY requires a first vertex element with unique x/y/z properties")
    if mode == 'ascii':
        values = np.loadtxt(stream, max_rows=number, ndmin=2)
        if values.shape != (number,len(props)):
            raise ValueError("PLY vertex count differs from header")
        columns = {name: values[:,i] for i,name in enumerate(names)}
    elif mode in ('binary_little_endian','binary_big_endian'):
        order = '<' if mode == 'binary_little_endian' else '>'
        dtype = np.dtype([(name,order+typ) for name,typ in props])
        data = stream.read(number*dtype.itemsize)
        if len(data) != number*dtype.itemsize:
            raise ValueError("truncated binary PLY")
        values = np.frombuffer(data,dtype=dtype,count=number)
        columns = {name: values[name] for name in names}
    else:
        raise ValueError("unsupported PLY encoding")
    xyz = np.column_stack([columns[axis] for axis in ('x','y','z')])
    colours = None
    if {'red','green','blue'} <= set(names):
        colours = np.column_stack([columns[k] for k in ('red','green','blue')])
    return xyz, colours


def load_cloud(path, max_points=50000):
    """Return finite raw XYZ, optional RGB, and original count; sample deterministically."""
    path = Path(path)
    if not isinstance(max_points,int) or max_points < 1:
        raise ValueError("max_points must be positive")
    if path.suffix.lower() == '.npy':
        values = np.load(path,allow_pickle=False,mmap_mode='r')
        if values.ndim != 2 or values.shape[1] not in (3,6) or not np.issubdtype(values.dtype,np.number):
            raise ValueError("NPY map must be numeric Nx3 XYZ or Nx6 XYZ/RGB")
        xyz, colours = values[:,:3], values[:,3:] if values.shape[1] == 6 else None
    else:
        with path.open('rb') as stream:
            if path.suffix.lower() == '.pcd':
                xyz,colours = _pcd(stream)
            elif path.suffix.lower() == '.ply':
                xyz,colours = _ply(stream)
            else:
                raise ValueError("supported point clouds: PCD, PLY and NPY")
    total = len(xyz)
    valid = np.isfinite(xyz).all(axis=1)
    xyz = np.asarray(xyz[valid],float)
    if not len(xyz):
        raise ValueError("map has no finite points")
    if colours is not None:
        colours = np.clip(np.nan_to_num(colours[valid]),0,255).astype(np.uint8)
    indices = np.linspace(0,len(xyz)-1,min(max_points,len(xyz)),dtype=int)
    return xyz[indices].copy(), colours[indices].copy() if colours is not None else None, total
