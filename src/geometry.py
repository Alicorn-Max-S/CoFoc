"""
3D Geometry and Mesh System for CoFoc Avatar

Provides mesh creation, geometry primitives, and GPU buffer management.
"""

from OpenGL.GL import *
import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from math3d import Vec3, Mat4


@dataclass
class Vertex:
    """Vertex with position, normal, UV, and bone data."""
    position: Vec3
    normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    uv: Tuple[float, float] = (0.0, 0.0)
    bone_weights: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    bone_indices: Tuple[int, int, int, int] = (0, 0, 0, 0)


class Mesh:
    """GPU mesh with VAO, VBO, and EBO."""

    def __init__(self):
        self.vao = 0
        self.vbo = 0
        self.ebo = 0
        self.index_count = 0
        self.vertex_count = 0
        self._initialized = False

    def upload(self, vertices: List[Vertex], indices: List[int]) -> None:
        """Upload mesh data to GPU."""
        if self._initialized:
            self.delete()

        self.vertex_count = len(vertices)
        self.index_count = len(indices)

        # Build vertex data array
        # Layout: position(3) + normal(3) + uv(2) + bone_weights(4) + bone_indices(4)
        vertex_data = []
        for v in vertices:
            vertex_data.extend([v.position.x, v.position.y, v.position.z])
            vertex_data.extend([v.normal.x, v.normal.y, v.normal.z])
            vertex_data.extend(v.uv)
            vertex_data.extend(v.bone_weights)
            # Store bone indices as floats for the array, will be cast to int in shader
            vertex_data.extend([float(i) for i in v.bone_indices])

        vertex_array = np.array(vertex_data, dtype=np.float32)
        index_array = np.array(indices, dtype=np.uint32)

        # Create VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Create VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array, GL_STATIC_DRAW)

        # Create EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_array.nbytes, index_array, GL_STATIC_DRAW)

        stride = 16 * 4  # 16 floats per vertex

        # Position attribute (location 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # Normal attribute (location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))

        # UV attribute (for eye shader, location 2 overlaps with bone weights conceptually)
        # We'll use bone_weights location for UV when needed
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8 * 4))

        # Bone indices attribute (location 3)
        glEnableVertexAttribArray(3)
        glVertexAttribIPointer(3, 4, GL_INT, stride, ctypes.c_void_p(12 * 4))

        glBindVertexArray(0)
        self._initialized = True

    def draw(self) -> None:
        """Draw the mesh."""
        if not self._initialized:
            return
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def delete(self) -> None:
        """Delete GPU resources."""
        if self._initialized:
            glDeleteVertexArrays(1, [self.vao])
            glDeleteBuffers(1, [self.vbo])
            glDeleteBuffers(1, [self.ebo])
            self.vao = 0
            self.vbo = 0
            self.ebo = 0
            self._initialized = False


class MeshBuilder:
    """Builder for constructing meshes from primitives."""

    def __init__(self):
        self.vertices: List[Vertex] = []
        self.indices: List[int] = []

    def clear(self) -> None:
        """Clear all geometry."""
        self.vertices.clear()
        self.indices.clear()

    def add_vertex(self, v: Vertex) -> int:
        """Add a vertex and return its index."""
        index = len(self.vertices)
        self.vertices.append(v)
        return index

    def add_triangle(self, i0: int, i1: int, i2: int) -> None:
        """Add a triangle from vertex indices."""
        self.indices.extend([i0, i1, i2])

    def add_quad(self, i0: int, i1: int, i2: int, i3: int) -> None:
        """Add a quad as two triangles."""
        self.indices.extend([i0, i1, i2, i0, i2, i3])

    def transform(self, matrix: Mat4) -> None:
        """Transform all vertices by a matrix."""
        for v in self.vertices:
            v.position = matrix.transform_point(v.position)
            v.normal = matrix.transform_direction(v.normal).normalized()

    def set_bone(self, bone_index: int, weight: float = 1.0) -> None:
        """Set bone assignment for all vertices."""
        for v in self.vertices:
            v.bone_indices = (bone_index, 0, 0, 0)
            v.bone_weights = (weight, 0.0, 0.0, 0.0)

    def set_bone_range(self, start: int, end: int, bone_index: int, weight: float = 1.0) -> None:
        """Set bone assignment for a range of vertices."""
        for i in range(start, min(end, len(self.vertices))):
            self.vertices[i].bone_indices = (bone_index, 0, 0, 0)
            self.vertices[i].bone_weights = (weight, 0.0, 0.0, 0.0)

    def build(self) -> Mesh:
        """Build and upload the mesh to GPU."""
        mesh = Mesh()
        mesh.upload(self.vertices, self.indices)
        return mesh


def create_sphere(radius: float = 1.0, segments: int = 16, rings: int = 12,
                  bone_index: int = 0) -> MeshBuilder:
    """Create a UV sphere."""
    builder = MeshBuilder()

    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        y = math.cos(phi) * radius
        ring_radius = math.sin(phi) * radius

        for seg in range(segments + 1):
            theta = 2 * math.pi * seg / segments
            x = math.cos(theta) * ring_radius
            z = math.sin(theta) * ring_radius

            normal = Vec3(x, y, z).normalized()
            u = seg / segments
            v = ring / rings

            vertex = Vertex(
                position=Vec3(x, y, z),
                normal=normal,
                uv=(u, v),
                bone_indices=(bone_index, 0, 0, 0),
                bone_weights=(1.0, 0.0, 0.0, 0.0)
            )
            builder.add_vertex(vertex)

    # Generate indices
    for ring in range(rings):
        for seg in range(segments):
            curr = ring * (segments + 1) + seg
            next_ring = curr + segments + 1

            builder.add_quad(curr, next_ring, next_ring + 1, curr + 1)

    return builder


def create_cylinder(radius: float = 1.0, height: float = 2.0, segments: int = 16,
                    bone_index: int = 0, cap_top: bool = True, cap_bottom: bool = True) -> MeshBuilder:
    """Create a cylinder."""
    builder = MeshBuilder()
    half_height = height / 2

    # Side vertices
    for i in range(2):
        y = half_height if i == 0 else -half_height
        for seg in range(segments + 1):
            theta = 2 * math.pi * seg / segments
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            normal = Vec3(x, 0, z).normalized()
            u = seg / segments
            v = i

            vertex = Vertex(
                position=Vec3(x, y, z),
                normal=normal,
                uv=(u, v),
                bone_indices=(bone_index, 0, 0, 0),
                bone_weights=(1.0, 0.0, 0.0, 0.0)
            )
            builder.add_vertex(vertex)

    # Side indices
    for seg in range(segments):
        top = seg
        bottom = seg + segments + 1
        builder.add_quad(top, top + 1, bottom + 1, bottom)

    # Top cap
    if cap_top:
        center_idx = len(builder.vertices)
        builder.add_vertex(Vertex(
            position=Vec3(0, half_height, 0),
            normal=Vec3(0, 1, 0),
            uv=(0.5, 0.5),
            bone_indices=(bone_index, 0, 0, 0),
            bone_weights=(1.0, 0.0, 0.0, 0.0)
        ))

        for seg in range(segments + 1):
            theta = 2 * math.pi * seg / segments
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            vertex = Vertex(
                position=Vec3(x, half_height, z),
                normal=Vec3(0, 1, 0),
                uv=(0.5 + math.cos(theta) * 0.5, 0.5 + math.sin(theta) * 0.5),
                bone_indices=(bone_index, 0, 0, 0),
                bone_weights=(1.0, 0.0, 0.0, 0.0)
            )
            builder.add_vertex(vertex)

        for seg in range(segments):
            builder.add_triangle(center_idx, center_idx + 1 + seg, center_idx + 2 + seg)

    # Bottom cap
    if cap_bottom:
        center_idx = len(builder.vertices)
        builder.add_vertex(Vertex(
            position=Vec3(0, -half_height, 0),
            normal=Vec3(0, -1, 0),
            uv=(0.5, 0.5),
            bone_indices=(bone_index, 0, 0, 0),
            bone_weights=(1.0, 0.0, 0.0, 0.0)
        ))

        for seg in range(segments + 1):
            theta = 2 * math.pi * seg / segments
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            vertex = Vertex(
                position=Vec3(x, -half_height, z),
                normal=Vec3(0, -1, 0),
                uv=(0.5 + math.cos(theta) * 0.5, 0.5 + math.sin(theta) * 0.5),
                bone_indices=(bone_index, 0, 0, 0),
                bone_weights=(1.0, 0.0, 0.0, 0.0)
            )
            builder.add_vertex(vertex)

        for seg in range(segments):
            builder.add_triangle(center_idx, center_idx + 2 + seg, center_idx + 1 + seg)

    return builder


def create_capsule(radius: float = 0.5, height: float = 2.0, segments: int = 16,
                   rings: int = 8, bone_index: int = 0) -> MeshBuilder:
    """Create a capsule (cylinder with hemisphere caps)."""
    builder = MeshBuilder()
    cylinder_height = height - 2 * radius

    # Top hemisphere
    for ring in range(rings // 2 + 1):
        phi = math.pi * ring / rings
        y = math.cos(phi) * radius + cylinder_height / 2
        ring_radius = math.sin(phi) * radius

        for seg in range(segments + 1):
            theta = 2 * math.pi * seg / segments
            x = math.cos(theta) * ring_radius
            z = math.sin(theta) * ring_radius

            normal = Vec3(math.cos(theta) * math.sin(phi),
                          math.cos(phi),
                          math.sin(theta) * math.sin(phi)).normalized()

            vertex = Vertex(
                position=Vec3(x, y, z),
                normal=normal,
                uv=(seg / segments, ring / rings),
                bone_indices=(bone_index, 0, 0, 0),
                bone_weights=(1.0, 0.0, 0.0, 0.0)
            )
            builder.add_vertex(vertex)

    # Cylinder section
    for i in range(2):
        y = (cylinder_height / 2) if i == 0 else (-cylinder_height / 2)
        for seg in range(segments + 1):
            theta = 2 * math.pi * seg / segments
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            normal = Vec3(x, 0, z).normalized()

            vertex = Vertex(
                position=Vec3(x, y, z),
                normal=normal,
                uv=(seg / segments, 0.5),
                bone_indices=(bone_index, 0, 0, 0),
                bone_weights=(1.0, 0.0, 0.0, 0.0)
            )
            builder.add_vertex(vertex)

    # Bottom hemisphere
    for ring in range(rings // 2, rings + 1):
        phi = math.pi * ring / rings
        y = math.cos(phi) * radius - cylinder_height / 2
        ring_radius = math.sin(phi) * radius

        for seg in range(segments + 1):
            theta = 2 * math.pi * seg / segments
            x = math.cos(theta) * ring_radius
            z = math.sin(theta) * ring_radius

            normal = Vec3(math.cos(theta) * math.sin(phi),
                          math.cos(phi),
                          math.sin(theta) * math.sin(phi)).normalized()

            vertex = Vertex(
                position=Vec3(x, y, z),
                normal=normal,
                uv=(seg / segments, ring / rings),
                bone_indices=(bone_index, 0, 0, 0),
                bone_weights=(1.0, 0.0, 0.0, 0.0)
            )
            builder.add_vertex(vertex)

    # Generate indices for hemispheres
    total_rings = rings + 3  # top hem + 2 cylinder + bottom hem
    for ring in range(total_rings):
        for seg in range(segments):
            curr = ring * (segments + 1) + seg
            next_ring = curr + segments + 1

            if next_ring + 1 < len(builder.vertices):
                builder.add_quad(curr, next_ring, next_ring + 1, curr + 1)

    return builder


def create_box(width: float = 1.0, height: float = 1.0, depth: float = 1.0,
               bone_index: int = 0) -> MeshBuilder:
    """Create a box."""
    builder = MeshBuilder()
    hw, hh, hd = width / 2, height / 2, depth / 2

    # Define face data: (normal, vertices)
    faces = [
        # Front face
        (Vec3(0, 0, 1), [
            (Vec3(-hw, -hh, hd), (0, 0)),
            (Vec3(hw, -hh, hd), (1, 0)),
            (Vec3(hw, hh, hd), (1, 1)),
            (Vec3(-hw, hh, hd), (0, 1)),
        ]),
        # Back face
        (Vec3(0, 0, -1), [
            (Vec3(hw, -hh, -hd), (0, 0)),
            (Vec3(-hw, -hh, -hd), (1, 0)),
            (Vec3(-hw, hh, -hd), (1, 1)),
            (Vec3(hw, hh, -hd), (0, 1)),
        ]),
        # Top face
        (Vec3(0, 1, 0), [
            (Vec3(-hw, hh, hd), (0, 0)),
            (Vec3(hw, hh, hd), (1, 0)),
            (Vec3(hw, hh, -hd), (1, 1)),
            (Vec3(-hw, hh, -hd), (0, 1)),
        ]),
        # Bottom face
        (Vec3(0, -1, 0), [
            (Vec3(-hw, -hh, -hd), (0, 0)),
            (Vec3(hw, -hh, -hd), (1, 0)),
            (Vec3(hw, -hh, hd), (1, 1)),
            (Vec3(-hw, -hh, hd), (0, 1)),
        ]),
        # Right face
        (Vec3(1, 0, 0), [
            (Vec3(hw, -hh, hd), (0, 0)),
            (Vec3(hw, -hh, -hd), (1, 0)),
            (Vec3(hw, hh, -hd), (1, 1)),
            (Vec3(hw, hh, hd), (0, 1)),
        ]),
        # Left face
        (Vec3(-1, 0, 0), [
            (Vec3(-hw, -hh, -hd), (0, 0)),
            (Vec3(-hw, -hh, hd), (1, 0)),
            (Vec3(-hw, hh, hd), (1, 1)),
            (Vec3(-hw, hh, -hd), (0, 1)),
        ]),
    ]

    for normal, verts in faces:
        start_idx = len(builder.vertices)
        for pos, uv in verts:
            vertex = Vertex(
                position=pos,
                normal=normal,
                uv=uv,
                bone_indices=(bone_index, 0, 0, 0),
                bone_weights=(1.0, 0.0, 0.0, 0.0)
            )
            builder.add_vertex(vertex)
        builder.add_quad(start_idx, start_idx + 1, start_idx + 2, start_idx + 3)

    return builder


def create_rounded_box(width: float = 1.0, height: float = 1.0, depth: float = 1.0,
                       radius: float = 0.1, segments: int = 4, bone_index: int = 0) -> MeshBuilder:
    """Create a rounded box (box with rounded edges)."""
    builder = MeshBuilder()

    # For simplicity, we'll create a regular box with smoothed normals
    # A full rounded box would require edge and corner geometry
    hw = (width - 2 * radius) / 2
    hh = (height - 2 * radius) / 2
    hd = (depth - 2 * radius) / 2

    # Create the 8 corner spheres
    corners = [
        Vec3(hw, hh, hd), Vec3(-hw, hh, hd),
        Vec3(hw, -hh, hd), Vec3(-hw, -hh, hd),
        Vec3(hw, hh, -hd), Vec3(-hw, hh, -hd),
        Vec3(hw, -hh, -hd), Vec3(-hw, -hh, -hd),
    ]

    # For each corner, add a small sphere
    for corner in corners:
        sphere = create_sphere(radius, segments, segments // 2, bone_index)
        sphere.transform(Mat4.translation(corner.x, corner.y, corner.z))
        offset = len(builder.vertices)
        builder.vertices.extend(sphere.vertices)
        for idx in sphere.indices:
            builder.indices.append(idx + offset)

    # Create the face plates
    # Front/Back faces
    for z_sign in [1, -1]:
        z = hd * z_sign
        normal = Vec3(0, 0, z_sign)
        start_idx = len(builder.vertices)

        for y_sign in [-1, 1]:
            for x_sign in [-1, 1]:
                x = hw * x_sign
                y = hh * y_sign
                vertex = Vertex(
                    position=Vec3(x, y, z + radius * z_sign),
                    normal=normal,
                    uv=(0.5 + x_sign * 0.5, 0.5 + y_sign * 0.5),
                    bone_indices=(bone_index, 0, 0, 0),
                    bone_weights=(1.0, 0.0, 0.0, 0.0)
                )
                builder.add_vertex(vertex)

        if z_sign > 0:
            builder.add_quad(start_idx, start_idx + 1, start_idx + 3, start_idx + 2)
        else:
            builder.add_quad(start_idx, start_idx + 2, start_idx + 3, start_idx + 1)

    # Top/Bottom faces
    for y_sign in [1, -1]:
        y = hh * y_sign
        normal = Vec3(0, y_sign, 0)
        start_idx = len(builder.vertices)

        for z_sign in [-1, 1]:
            for x_sign in [-1, 1]:
                x = hw * x_sign
                z = hd * z_sign
                vertex = Vertex(
                    position=Vec3(x, y + radius * y_sign, z),
                    normal=normal,
                    uv=(0.5 + x_sign * 0.5, 0.5 + z_sign * 0.5),
                    bone_indices=(bone_index, 0, 0, 0),
                    bone_weights=(1.0, 0.0, 0.0, 0.0)
                )
                builder.add_vertex(vertex)

        if y_sign > 0:
            builder.add_quad(start_idx, start_idx + 2, start_idx + 3, start_idx + 1)
        else:
            builder.add_quad(start_idx, start_idx + 1, start_idx + 3, start_idx + 2)

    # Left/Right faces
    for x_sign in [1, -1]:
        x = hw * x_sign
        normal = Vec3(x_sign, 0, 0)
        start_idx = len(builder.vertices)

        for z_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                y = hh * y_sign
                z = hd * z_sign
                vertex = Vertex(
                    position=Vec3(x + radius * x_sign, y, z),
                    normal=normal,
                    uv=(0.5 + z_sign * 0.5, 0.5 + y_sign * 0.5),
                    bone_indices=(bone_index, 0, 0, 0),
                    bone_weights=(1.0, 0.0, 0.0, 0.0)
                )
                builder.add_vertex(vertex)

        if x_sign > 0:
            builder.add_quad(start_idx, start_idx + 1, start_idx + 3, start_idx + 2)
        else:
            builder.add_quad(start_idx, start_idx + 2, start_idx + 3, start_idx + 1)

    return builder


def create_torus(major_radius: float = 1.0, minor_radius: float = 0.3,
                 major_segments: int = 24, minor_segments: int = 12,
                 bone_index: int = 0) -> MeshBuilder:
    """Create a torus (donut shape)."""
    builder = MeshBuilder()

    for i in range(major_segments + 1):
        theta = 2 * math.pi * i / major_segments

        for j in range(minor_segments + 1):
            phi = 2 * math.pi * j / minor_segments

            x = (major_radius + minor_radius * math.cos(phi)) * math.cos(theta)
            y = minor_radius * math.sin(phi)
            z = (major_radius + minor_radius * math.cos(phi)) * math.sin(theta)

            # Normal
            nx = math.cos(phi) * math.cos(theta)
            ny = math.sin(phi)
            nz = math.cos(phi) * math.sin(theta)

            vertex = Vertex(
                position=Vec3(x, y, z),
                normal=Vec3(nx, ny, nz),
                uv=(i / major_segments, j / minor_segments),
                bone_indices=(bone_index, 0, 0, 0),
                bone_weights=(1.0, 0.0, 0.0, 0.0)
            )
            builder.add_vertex(vertex)

    for i in range(major_segments):
        for j in range(minor_segments):
            curr = i * (minor_segments + 1) + j
            next_major = curr + minor_segments + 1

            builder.add_quad(curr, next_major, next_major + 1, curr + 1)

    return builder


def create_plane(width: float = 1.0, height: float = 1.0, segments_x: int = 1,
                 segments_y: int = 1, bone_index: int = 0) -> MeshBuilder:
    """Create a plane facing up (Y+)."""
    builder = MeshBuilder()

    for y in range(segments_y + 1):
        for x in range(segments_x + 1):
            u = x / segments_x
            v = y / segments_y
            px = (u - 0.5) * width
            pz = (v - 0.5) * height

            vertex = Vertex(
                position=Vec3(px, 0, pz),
                normal=Vec3(0, 1, 0),
                uv=(u, v),
                bone_indices=(bone_index, 0, 0, 0),
                bone_weights=(1.0, 0.0, 0.0, 0.0)
            )
            builder.add_vertex(vertex)

    for y in range(segments_y):
        for x in range(segments_x):
            curr = y * (segments_x + 1) + x
            next_row = curr + segments_x + 1

            builder.add_quad(curr, next_row, next_row + 1, curr + 1)

    return builder


def merge_builders(*builders: MeshBuilder) -> MeshBuilder:
    """Merge multiple mesh builders into one."""
    result = MeshBuilder()

    for builder in builders:
        offset = len(result.vertices)
        result.vertices.extend(builder.vertices)
        for idx in builder.indices:
            result.indices.append(idx + offset)

    return result
