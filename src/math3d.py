"""
3D Mathematics Utilities for CoFoc Avatar

Provides vector, matrix, and quaternion operations for 3D transformations.
"""

import numpy as np
import math
from typing import Tuple, Union


class Vec3:
    """3D Vector class with common operations."""

    __slots__ = ['x', 'y', 'z']

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vec3':
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Vec3':
        return self.__mul__(scalar)

    def __neg__(self) -> 'Vec3':
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: 'Vec3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'Vec3':
        l = self.length()
        if l < 1e-10:
            return Vec3(0, 0, 0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def lerp(self, other: 'Vec3', t: float) -> 'Vec3':
        """Linear interpolation between this vector and another."""
        return Vec3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t
        )

    @staticmethod
    def from_array(arr) -> 'Vec3':
        return Vec3(arr[0], arr[1], arr[2])

    @staticmethod
    def up() -> 'Vec3':
        return Vec3(0, 1, 0)

    @staticmethod
    def forward() -> 'Vec3':
        return Vec3(0, 0, -1)

    @staticmethod
    def right() -> 'Vec3':
        return Vec3(1, 0, 0)


class Quaternion:
    """Quaternion class for rotation representation."""

    __slots__ = ['w', 'x', 'y', 'z']

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Quaternion({self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication (rotation composition)."""
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )

    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def length(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'Quaternion':
        l = self.length()
        if l < 1e-10:
            return Quaternion()
        return Quaternion(self.w / l, self.x / l, self.y / l, self.z / l)

    def rotate_vector(self, v: Vec3) -> Vec3:
        """Rotate a vector by this quaternion."""
        qv = Quaternion(0, v.x, v.y, v.z)
        result = self * qv * self.conjugate()
        return Vec3(result.x, result.y, result.z)

    def to_matrix4(self) -> np.ndarray:
        """Convert quaternion to 4x4 rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z

        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        return np.array([
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy), 0],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx), 0],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def to_euler(self) -> Vec3:
        """Convert to Euler angles (pitch, yaw, roll) in radians."""
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return Vec3(pitch, yaw, roll)

    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> 'Quaternion':
        """Create quaternion from axis-angle representation."""
        axis = axis.normalized()
        half_angle = angle / 2
        s = math.sin(half_angle)
        return Quaternion(
            math.cos(half_angle),
            axis.x * s,
            axis.y * s,
            axis.z * s
        )

    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float) -> 'Quaternion':
        """Create quaternion from Euler angles (in radians)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        )

    @staticmethod
    def slerp(q1: 'Quaternion', q2: 'Quaternion', t: float) -> 'Quaternion':
        """Spherical linear interpolation between two quaternions."""
        dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

        if dot < 0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot

        if dot > 0.9995:
            result = Quaternion(
                q1.w + t * (q2.w - q1.w),
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z)
            )
            return result.normalized()

        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return Quaternion(
            s0 * q1.w + s1 * q2.w,
            s0 * q1.x + s1 * q2.x,
            s0 * q1.y + s1 * q2.y,
            s0 * q1.z + s1 * q2.z
        )

    @staticmethod
    def identity() -> 'Quaternion':
        return Quaternion(1, 0, 0, 0)


class Mat4:
    """4x4 Matrix class for 3D transformations."""

    def __init__(self, data: np.ndarray = None):
        if data is None:
            self.data = np.eye(4, dtype=np.float32)
        else:
            self.data = np.array(data, dtype=np.float32).reshape(4, 4)

    def __repr__(self):
        return f"Mat4(\n{self.data}\n)"

    def __matmul__(self, other: 'Mat4') -> 'Mat4':
        """Matrix multiplication."""
        return Mat4(self.data @ other.data)

    def transform_point(self, v: Vec3) -> Vec3:
        """Transform a point by this matrix."""
        p = np.array([v.x, v.y, v.z, 1.0], dtype=np.float32)
        result = self.data @ p
        if abs(result[3]) > 1e-10:
            result /= result[3]
        return Vec3(result[0], result[1], result[2])

    def transform_direction(self, v: Vec3) -> Vec3:
        """Transform a direction vector (ignores translation)."""
        d = np.array([v.x, v.y, v.z, 0.0], dtype=np.float32)
        result = self.data @ d
        return Vec3(result[0], result[1], result[2])

    def to_array(self) -> np.ndarray:
        """Return as column-major array for OpenGL."""
        return self.data.T.flatten().astype(np.float32)

    def inverse(self) -> 'Mat4':
        """Return the inverse of this matrix."""
        return Mat4(np.linalg.inv(self.data))

    def transposed(self) -> 'Mat4':
        """Return the transpose of this matrix."""
        return Mat4(self.data.T)

    @staticmethod
    def identity() -> 'Mat4':
        return Mat4()

    @staticmethod
    def translation(x: float, y: float, z: float) -> 'Mat4':
        m = np.eye(4, dtype=np.float32)
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
        return Mat4(m)

    @staticmethod
    def scale(x: float, y: float = None, z: float = None) -> 'Mat4':
        if y is None:
            y = x
        if z is None:
            z = x
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = x
        m[1, 1] = y
        m[2, 2] = z
        return Mat4(m)

    @staticmethod
    def rotation_x(angle: float) -> 'Mat4':
        c, s = math.cos(angle), math.sin(angle)
        return Mat4(np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32))

    @staticmethod
    def rotation_y(angle: float) -> 'Mat4':
        c, s = math.cos(angle), math.sin(angle)
        return Mat4(np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32))

    @staticmethod
    def rotation_z(angle: float) -> 'Mat4':
        c, s = math.cos(angle), math.sin(angle)
        return Mat4(np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32))

    @staticmethod
    def from_quaternion(q: Quaternion) -> 'Mat4':
        return Mat4(q.to_matrix4())

    @staticmethod
    def perspective(fov: float, aspect: float, near: float, far: float) -> 'Mat4':
        """Create a perspective projection matrix."""
        f = 1.0 / math.tan(fov / 2)
        return Mat4(np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32))

    @staticmethod
    def orthographic(left: float, right: float, bottom: float, top: float,
                     near: float, far: float) -> 'Mat4':
        """Create an orthographic projection matrix."""
        return Mat4(np.array([
            [2 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1]
        ], dtype=np.float32))

    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3) -> 'Mat4':
        """Create a view matrix looking from eye toward target."""
        forward = (target - eye).normalized()
        right = forward.cross(up).normalized()
        up = right.cross(forward)

        return Mat4(np.array([
            [right.x, right.y, right.z, -right.dot(eye)],
            [up.x, up.y, up.z, -up.dot(eye)],
            [-forward.x, -forward.y, -forward.z, forward.dot(eye)],
            [0, 0, 0, 1]
        ], dtype=np.float32))


class Transform:
    """Complete 3D transformation with position, rotation, and scale."""

    def __init__(self):
        self.position = Vec3(0, 0, 0)
        self.rotation = Quaternion.identity()
        self.scale_factor = Vec3(1, 1, 1)
        self._matrix_dirty = True
        self._cached_matrix = None

    def set_position(self, x: float, y: float, z: float):
        self.position = Vec3(x, y, z)
        self._matrix_dirty = True

    def set_rotation(self, q: Quaternion):
        self.rotation = q
        self._matrix_dirty = True

    def set_euler(self, pitch: float, yaw: float, roll: float):
        self.rotation = Quaternion.from_euler(pitch, yaw, roll)
        self._matrix_dirty = True

    def set_scale(self, x: float, y: float = None, z: float = None):
        if y is None:
            y = x
        if z is None:
            z = x
        self.scale_factor = Vec3(x, y, z)
        self._matrix_dirty = True

    def rotate_around_axis(self, axis: Vec3, angle: float):
        """Rotate this transform around an axis."""
        q = Quaternion.from_axis_angle(axis, angle)
        self.rotation = q * self.rotation
        self._matrix_dirty = True

    def to_matrix(self) -> Mat4:
        """Get the combined transformation matrix."""
        if self._matrix_dirty:
            t = Mat4.translation(self.position.x, self.position.y, self.position.z)
            r = Mat4.from_quaternion(self.rotation)
            s = Mat4.scale(self.scale_factor.x, self.scale_factor.y, self.scale_factor.z)
            self._cached_matrix = t @ r @ s
            self._matrix_dirty = False
        return self._cached_matrix

    def forward(self) -> Vec3:
        """Get the forward direction of this transform."""
        return self.rotation.rotate_vector(Vec3(0, 0, -1))

    def right(self) -> Vec3:
        """Get the right direction of this transform."""
        return self.rotation.rotate_vector(Vec3(1, 0, 0))

    def up(self) -> Vec3:
        """Get the up direction of this transform."""
        return self.rotation.rotate_vector(Vec3(0, 1, 0))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values."""
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth interpolation."""
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3 - 2 * t)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def degrees(radians_val: float) -> float:
    """Convert radians to degrees."""
    return radians_val * 180.0 / math.pi
