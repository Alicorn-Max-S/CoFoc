"""
glTF/GLB/VRM Model Loader for CoFoc

Loads 3D models from glTF 2.0 format files including GLB binaries and VRM avatars.
"""

import os
import struct
import json
import base64
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from io import BytesIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from math3d import Vec3, Quaternion, Mat4, Transform
from geometry import Mesh, MeshBuilder, Vertex
from animation import Skeleton, Bone, AnimationClip, BoneAnimation, Keyframe


@dataclass
class GLTFMaterial:
    """Material extracted from glTF."""
    name: str = "default"
    base_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    texture_id: Optional[int] = None


@dataclass
class GLTFPrimitive:
    """A single primitive within a mesh."""
    vertices: List[Vertex] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    material_index: int = 0


@dataclass
class GLTFMesh:
    """Mesh extracted from glTF."""
    name: str = "mesh"
    primitives: List[GLTFPrimitive] = field(default_factory=list)


@dataclass
class GLTFNode:
    """Node in the glTF scene graph."""
    name: str = "node"
    mesh_index: Optional[int] = None
    skin_index: Optional[int] = None
    children: List[int] = field(default_factory=list)
    translation: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    scale: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))

    def get_local_matrix(self) -> Mat4:
        """Get the local transformation matrix."""
        t = Mat4.translation(self.translation.x, self.translation.y, self.translation.z)
        r = Mat4.from_quaternion(self.rotation)
        s = Mat4.scale(self.scale.x, self.scale.y, self.scale.z)
        return t @ r @ s


@dataclass
class GLTFSkin:
    """Skin (skeleton) data from glTF."""
    name: str = "skin"
    joints: List[int] = field(default_factory=list)  # Node indices
    inverse_bind_matrices: List[Mat4] = field(default_factory=list)
    skeleton_root: Optional[int] = None


class GLTFModel:
    """Complete loaded glTF model."""

    def __init__(self):
        self.meshes: List[GLTFMesh] = []
        self.materials: List[GLTFMaterial] = []
        self.nodes: List[GLTFNode] = []
        self.skins: List[GLTFSkin] = []
        self.animations: List[AnimationClip] = []
        self.scene_nodes: List[int] = []  # Root nodes of the default scene


class GLTFLoader:
    """Loader for glTF 2.0 files (including GLB and VRM)."""

    COMPONENT_TYPES = {
        5120: ('b', 1),  # BYTE
        5121: ('B', 1),  # UNSIGNED_BYTE
        5122: ('h', 2),  # SHORT
        5123: ('H', 2),  # UNSIGNED_SHORT
        5125: ('I', 4),  # UNSIGNED_INT
        5126: ('f', 4),  # FLOAT
    }

    ACCESSOR_TYPES = {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16,
    }

    def __init__(self):
        self.gltf: Dict = {}
        self.buffers: List[bytes] = []
        self.base_path: Path = Path(".")

    def load(self, path: str) -> GLTFModel:
        """Load a glTF/GLB/VRM file."""
        path = Path(path)
        self.base_path = path.parent

        if path.suffix.lower() in ['.glb', '.vrm']:
            self._load_glb(path)
        else:
            self._load_gltf(path)

        return self._parse_model()

    def load_from_bytes(self, data: bytes, is_binary: bool = True) -> GLTFModel:
        """Load a glTF model from bytes."""
        if is_binary:
            self._parse_glb(data)
        else:
            self.gltf = json.loads(data.decode('utf-8'))
            self._load_external_buffers()

        return self._parse_model()

    def _load_glb(self, path: Path) -> None:
        """Load a GLB binary file."""
        with open(path, 'rb') as f:
            data = f.read()
        self._parse_glb(data)

    def _parse_glb(self, data: bytes) -> None:
        """Parse GLB binary data."""
        # GLB header: magic (4) + version (4) + length (4)
        magic, version, length = struct.unpack('<III', data[:12])

        if magic != 0x46546C67:  # 'glTF'
            raise ValueError("Invalid GLB file: bad magic number")

        if version != 2:
            raise ValueError(f"Unsupported glTF version: {version}")

        offset = 12
        while offset < length:
            chunk_length, chunk_type = struct.unpack('<II', data[offset:offset + 8])
            offset += 8

            chunk_data = data[offset:offset + chunk_length]
            offset += chunk_length

            if chunk_type == 0x4E4F534A:  # 'JSON'
                self.gltf = json.loads(chunk_data.decode('utf-8'))
            elif chunk_type == 0x004E4942:  # 'BIN'
                self.buffers.append(chunk_data)

    def _load_gltf(self, path: Path) -> None:
        """Load a JSON glTF file."""
        with open(path, 'r') as f:
            self.gltf = json.load(f)
        self._load_external_buffers()

    def _load_external_buffers(self) -> None:
        """Load external buffer files referenced by glTF."""
        for buffer_info in self.gltf.get('buffers', []):
            uri = buffer_info.get('uri', '')

            if uri.startswith('data:'):
                # Base64 encoded data
                _, data = uri.split(',', 1)
                self.buffers.append(base64.b64decode(data))
            elif uri:
                # External file
                buffer_path = self.base_path / uri
                with open(buffer_path, 'rb') as f:
                    self.buffers.append(f.read())

    def _get_accessor_data(self, accessor_index: int) -> np.ndarray:
        """Get data from an accessor."""
        accessor = self.gltf['accessors'][accessor_index]
        buffer_view = self.gltf['bufferViews'][accessor['bufferView']]

        buffer_data = self.buffers[buffer_view.get('buffer', 0)]
        byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        byte_length = buffer_view['byteLength']

        component_type = accessor['componentType']
        accessor_type = accessor['type']
        count = accessor['count']

        fmt, size = self.COMPONENT_TYPES[component_type]
        num_components = self.ACCESSOR_TYPES[accessor_type]

        # Handle byte stride
        byte_stride = buffer_view.get('byteStride', size * num_components)

        data = []
        for i in range(count):
            offset = byte_offset + i * byte_stride
            values = struct.unpack(f'<{num_components}{fmt}',
                                   buffer_data[offset:offset + size * num_components])
            data.append(values if num_components > 1 else values[0])

        return np.array(data, dtype=np.float32 if fmt == 'f' else np.int32)

    def _parse_model(self) -> GLTFModel:
        """Parse the loaded glTF data into a model."""
        model = GLTFModel()

        # Parse materials
        for mat_data in self.gltf.get('materials', []):
            material = self._parse_material(mat_data)
            model.materials.append(material)

        # Add default material if none exist
        if not model.materials:
            model.materials.append(GLTFMaterial())

        # Parse meshes
        for mesh_data in self.gltf.get('meshes', []):
            mesh = self._parse_mesh(mesh_data)
            model.meshes.append(mesh)

        # Parse nodes
        for node_data in self.gltf.get('nodes', []):
            node = self._parse_node(node_data)
            model.nodes.append(node)

        # Parse skins (skeletons)
        for skin_data in self.gltf.get('skins', []):
            skin = self._parse_skin(skin_data)
            model.skins.append(skin)

        # Parse animations
        for anim_data in self.gltf.get('animations', []):
            anim = self._parse_animation(anim_data, model.nodes)
            model.animations.append(anim)

        # Get default scene nodes
        default_scene = self.gltf.get('scene', 0)
        if 'scenes' in self.gltf and self.gltf['scenes']:
            scene = self.gltf['scenes'][default_scene]
            model.scene_nodes = scene.get('nodes', [])

        return model

    def _parse_material(self, mat_data: Dict) -> GLTFMaterial:
        """Parse material data."""
        material = GLTFMaterial(name=mat_data.get('name', 'material'))

        pbr = mat_data.get('pbrMetallicRoughness', {})
        base_color = pbr.get('baseColorFactor', [0.8, 0.8, 0.8, 1.0])
        material.base_color = tuple(base_color)
        material.metallic = pbr.get('metallicFactor', 0.0)
        material.roughness = pbr.get('roughnessFactor', 0.5)

        emissive = mat_data.get('emissiveFactor', [0.0, 0.0, 0.0])
        material.emissive = tuple(emissive)

        if 'baseColorTexture' in pbr:
            material.texture_id = pbr['baseColorTexture'].get('index')

        return material

    def _parse_mesh(self, mesh_data: Dict) -> GLTFMesh:
        """Parse mesh data."""
        mesh = GLTFMesh(name=mesh_data.get('name', 'mesh'))

        for prim_data in mesh_data.get('primitives', []):
            primitive = self._parse_primitive(prim_data)
            mesh.primitives.append(primitive)

        return mesh

    def _parse_primitive(self, prim_data: Dict) -> GLTFPrimitive:
        """Parse primitive data."""
        primitive = GLTFPrimitive()
        primitive.material_index = prim_data.get('material', 0)

        attributes = prim_data.get('attributes', {})

        # Get position data (required)
        positions = self._get_accessor_data(attributes['POSITION'])

        # Get optional attributes
        normals = None
        if 'NORMAL' in attributes:
            normals = self._get_accessor_data(attributes['NORMAL'])

        texcoords = None
        if 'TEXCOORD_0' in attributes:
            texcoords = self._get_accessor_data(attributes['TEXCOORD_0'])

        joints = None
        if 'JOINTS_0' in attributes:
            joints = self._get_accessor_data(attributes['JOINTS_0'])

        weights = None
        if 'WEIGHTS_0' in attributes:
            weights = self._get_accessor_data(attributes['WEIGHTS_0'])

        # Build vertices
        for i in range(len(positions)):
            pos = Vec3(positions[i][0], positions[i][1], positions[i][2])

            normal = Vec3(0, 1, 0)
            if normals is not None:
                normal = Vec3(normals[i][0], normals[i][1], normals[i][2])

            uv = (0.0, 0.0)
            if texcoords is not None:
                uv = (texcoords[i][0], texcoords[i][1])

            bone_indices = (0, 0, 0, 0)
            bone_weights = (1.0, 0.0, 0.0, 0.0)
            if joints is not None and weights is not None:
                bone_indices = tuple(int(j) for j in joints[i])
                bone_weights = tuple(float(w) for w in weights[i])

            vertex = Vertex(
                position=pos,
                normal=normal,
                uv=uv,
                bone_indices=bone_indices,
                bone_weights=bone_weights
            )
            primitive.vertices.append(vertex)

        # Get indices
        if 'indices' in prim_data:
            indices = self._get_accessor_data(prim_data['indices'])
            primitive.indices = [int(i) for i in indices]
        else:
            # Generate indices for non-indexed geometry
            primitive.indices = list(range(len(positions)))

        return primitive

    def _parse_node(self, node_data: Dict) -> GLTFNode:
        """Parse node data."""
        node = GLTFNode(name=node_data.get('name', 'node'))

        node.mesh_index = node_data.get('mesh')
        node.skin_index = node_data.get('skin')
        node.children = node_data.get('children', [])

        if 'translation' in node_data:
            t = node_data['translation']
            node.translation = Vec3(t[0], t[1], t[2])

        if 'rotation' in node_data:
            r = node_data['rotation']
            node.rotation = Quaternion(r[3], r[0], r[1], r[2])  # glTF uses xyzw

        if 'scale' in node_data:
            s = node_data['scale']
            node.scale = Vec3(s[0], s[1], s[2])

        if 'matrix' in node_data:
            # Decompose matrix
            m = node_data['matrix']
            # glTF uses column-major, but our Mat4 constructor expects row-major input
            # if we pass the array as is.
            # np.reshape fills row by row.
            # To get column-major reconstruction, we reshape then transpose.
            # Mat4 wraps the numpy array, which is now in correct layout (M[row, col])
            mat_data = np.array(m).reshape(4, 4).T
            mat = Mat4(mat_data)

            t, r, s = mat.decompose()
            node.translation = t
            node.rotation = r
            node.scale = s

        return node

    def _parse_skin(self, skin_data: Dict) -> GLTFSkin:
        """Parse skin (skeleton) data."""
        skin = GLTFSkin(name=skin_data.get('name', 'skin'))
        skin.joints = skin_data.get('joints', [])
        skin.skeleton_root = skin_data.get('skeleton')

        if 'inverseBindMatrices' in skin_data:
            ibm_data = self._get_accessor_data(skin_data['inverseBindMatrices'])
            for i in range(len(skin.joints)):
                mat_data = ibm_data[i].reshape(4, 4).T  # Column-major to row-major
                skin.inverse_bind_matrices.append(Mat4(mat_data))

        return skin

    def _parse_animation(self, anim_data: Dict, nodes: List[GLTFNode]) -> AnimationClip:
        """Parse animation data."""
        clip = AnimationClip(
            name=anim_data.get('name', 'animation'),
            duration=0.0,
            looping=True
        )

        samplers = anim_data.get('samplers', [])

        for channel in anim_data.get('channels', []):
            sampler_index = channel['sampler']
            target = channel['target']

            if 'node' not in target:
                continue

            node_index = target['node']
            path = target['path']

            if node_index >= len(nodes):
                continue

            node_name = nodes[node_index].name

            sampler = samplers[sampler_index]
            times = self._get_accessor_data(sampler['input'])
            values = self._get_accessor_data(sampler['output'])

            clip.duration = max(clip.duration, float(times[-1]))

            if node_name not in clip.bone_animations:
                clip.bone_animations[node_name] = BoneAnimation(node_name)

            bone_anim = clip.bone_animations[node_name]

            for i, t in enumerate(times):
                # Find or create keyframe at this time
                keyframe = None
                for kf in bone_anim.keyframes:
                    if abs(kf.time - t) < 0.001:
                        keyframe = kf
                        break
                if keyframe is None:
                    keyframe = Keyframe(time=float(t))
                    bone_anim.keyframes.append(keyframe)

                if path == 'translation':
                    keyframe.position = Vec3(values[i][0], values[i][1], values[i][2])
                elif path == 'rotation':
                    keyframe.rotation = Quaternion(values[i][3], values[i][0],
                                                   values[i][1], values[i][2])
                elif path == 'scale':
                    keyframe.scale = Vec3(values[i][0], values[i][1], values[i][2])

            # Sort keyframes by time
            bone_anim.keyframes.sort(key=lambda kf: kf.time)

        return clip


def download_ready_player_me_avatar(avatar_id: str, output_path: str,
                                    quality: str = "high") -> bool:
    """
    Download an avatar from Ready Player Me.

    Args:
        avatar_id: The avatar ID (e.g., "6185a4acfb622cf1cdc49348")
        output_path: Where to save the GLB file
        quality: "high", "medium", or "low"

    Returns:
        True if successful, False otherwise
    """
    import requests

    quality_map = {"high": 0, "medium": 1, "low": 2}
    mesh_lod = quality_map.get(quality, 1)

    url = f"https://models.readyplayer.me/{avatar_id}.glb"
    params = {
        "meshLod": mesh_lod,
        "pose": "A",  # A-pose for animation
        "morphTargets": "ARKit,Oculus Visemes",  # Include blend shapes
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"Downloaded avatar to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to download avatar: {e}")
        return False


def build_skeleton_from_gltf(model: GLTFModel, skin_index: int = 0) -> Skeleton:
    """Build a Skeleton from glTF skin data."""
    skeleton = Skeleton()

    if not model.skins or skin_index >= len(model.skins):
        return skeleton

    skin = model.skins[skin_index]

    # Map joint node indices to bone indices
    joint_to_bone: Dict[int, int] = {}

    for i, joint_node_index in enumerate(skin.joints):
        node = model.nodes[joint_node_index]

        # Find parent
        parent_name = None
        for potential_parent_idx in skin.joints[:i]:
            potential_parent = model.nodes[potential_parent_idx]
            if joint_node_index in potential_parent.children:
                parent_name = potential_parent.name
                break

        bone = skeleton.add_bone(node.name, parent_name)
        bone.local_bind_transform.position = node.translation
        bone.local_bind_transform.rotation = node.rotation
        bone.local_bind_transform.scale_factor = node.scale

        if i < len(skin.inverse_bind_matrices):
            bone.inverse_bind_matrix = skin.inverse_bind_matrices[i]

        joint_to_bone[joint_node_index] = bone.index

    return skeleton


def build_meshes_from_gltf(model: GLTFModel) -> List[Tuple[Mesh, GLTFMaterial]]:
    """Build GPU meshes from glTF model data."""
    result = []

    for mesh in model.meshes:
        for primitive in mesh.primitives:
            gpu_mesh = Mesh()
            gpu_mesh.upload(primitive.vertices, primitive.indices)

            material = model.materials[primitive.material_index] if primitive.material_index < len(
                model.materials) else model.materials[0]

            result.append((gpu_mesh, material))

    return result
