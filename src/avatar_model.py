"""
3D Humanoid Avatar Model for CoFoc

Constructs a stylized humanoid character with skeleton and mesh.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from math3d import Vec3, Quaternion, Mat4, Transform, radians
from geometry import (
    Mesh, MeshBuilder, Vertex, create_sphere, create_cylinder,
    create_capsule, create_box, merge_builders
)
from animation import Skeleton, Bone, AnimationController, ProceduralAnimator
from animation import create_idle_animation, create_speaking_animation, create_wave_animation


@dataclass
class AvatarColors:
    """Color scheme for the avatar."""
    skin: Tuple[float, float, float] = (0.95, 0.85, 0.75)
    body: Tuple[float, float, float] = (0.3, 0.4, 0.5)
    body_accent: Tuple[float, float, float] = (0.4, 0.5, 0.6)
    eye_white: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    eye_iris: Tuple[float, float, float] = (0.2, 0.4, 0.6)
    eye_pupil: Tuple[float, float, float] = (0.05, 0.05, 0.1)
    mouth_inner: Tuple[float, float, float] = (0.3, 0.15, 0.15)
    hair: Tuple[float, float, float] = (0.15, 0.1, 0.08)


class AvatarBodyPart:
    """A body part with mesh and material."""

    def __init__(self, name: str, mesh: Mesh, bone_index: int = 0):
        self.name = name
        self.mesh = mesh
        self.bone_index = bone_index
        self.diffuse_color = (0.8, 0.8, 0.8)
        self.specular_color = (0.3, 0.3, 0.3)
        self.shininess = 32.0
        self.emission = 0.0
        self.visible = True

    def set_color(self, r: float, g: float, b: float) -> 'AvatarBodyPart':
        self.diffuse_color = (r, g, b)
        return self

    def set_specular(self, r: float, g: float, b: float, shininess: float = 32.0) -> 'AvatarBodyPart':
        self.specular_color = (r, g, b)
        self.shininess = shininess
        return self


class HumanoidAvatar:
    """Complete 3D humanoid avatar with skeleton and meshes."""

    # Bone indices
    BONE_ROOT = 0
    BONE_HIPS = 1
    BONE_SPINE = 2
    BONE_CHEST = 3
    BONE_NECK = 4
    BONE_HEAD = 5
    BONE_JAW = 6
    BONE_SHOULDER_L = 7
    BONE_UPPER_ARM_L = 8
    BONE_FOREARM_L = 9
    BONE_HAND_L = 10
    BONE_SHOULDER_R = 11
    BONE_UPPER_ARM_R = 12
    BONE_FOREARM_R = 13
    BONE_HAND_R = 14
    BONE_EYE_L = 15
    BONE_EYE_R = 16

    def __init__(self):
        self.skeleton = Skeleton()
        self.body_parts: List[AvatarBodyPart] = []
        self.colors = AvatarColors()
        self.animation_controller: Optional[AnimationController] = None
        self.procedural_animator: Optional[ProceduralAnimator] = None

        # Eye state
        self.blink_amount = 0.0
        self.pupil_size = 0.3
        self.look_direction = (0.0, 0.0)

        self._build_skeleton()
        self._build_meshes()
        self._setup_animations()

    def _build_skeleton(self) -> None:
        """Build the humanoid skeleton hierarchy."""
        # Root and spine
        root = self.skeleton.add_bone("root")
        root.local_bind_transform.set_position(0, 0, 0)

        hips = self.skeleton.add_bone("hips", "root")
        hips.local_bind_transform.set_position(0, 0.9, 0)

        spine = self.skeleton.add_bone("spine", "hips")
        spine.local_bind_transform.set_position(0, 0.15, 0)

        chest = self.skeleton.add_bone("chest", "spine")
        chest.local_bind_transform.set_position(0, 0.25, 0)

        neck = self.skeleton.add_bone("neck", "chest")
        neck.local_bind_transform.set_position(0, 0.2, 0)

        head = self.skeleton.add_bone("head", "neck")
        head.local_bind_transform.set_position(0, 0.1, 0)

        jaw = self.skeleton.add_bone("jaw", "head")
        jaw.local_bind_transform.set_position(0, -0.05, 0.08)

        # Left arm
        shoulder_l = self.skeleton.add_bone("shoulder_l", "chest")
        shoulder_l.local_bind_transform.set_position(-0.15, 0.15, 0)

        upper_arm_l = self.skeleton.add_bone("upper_arm_l", "shoulder_l")
        upper_arm_l.local_bind_transform.set_position(-0.08, -0.02, 0)
        upper_arm_l.local_bind_transform.set_euler(0, 0, radians(15))

        forearm_l = self.skeleton.add_bone("forearm_l", "upper_arm_l")
        forearm_l.local_bind_transform.set_position(0, -0.25, 0)

        hand_l = self.skeleton.add_bone("hand_l", "forearm_l")
        hand_l.local_bind_transform.set_position(0, -0.22, 0)

        # Right arm
        shoulder_r = self.skeleton.add_bone("shoulder_r", "chest")
        shoulder_r.local_bind_transform.set_position(0.15, 0.15, 0)

        upper_arm_r = self.skeleton.add_bone("upper_arm_r", "shoulder_r")
        upper_arm_r.local_bind_transform.set_position(0.08, -0.02, 0)
        upper_arm_r.local_bind_transform.set_euler(0, 0, radians(-15))

        forearm_r = self.skeleton.add_bone("forearm_r", "upper_arm_r")
        forearm_r.local_bind_transform.set_position(0, -0.25, 0)

        hand_r = self.skeleton.add_bone("hand_r", "forearm_r")
        hand_r.local_bind_transform.set_position(0, -0.22, 0)

        # Eyes
        eye_l = self.skeleton.add_bone("eye_l", "head")
        eye_l.local_bind_transform.set_position(-0.06, 0.08, 0.12)

        eye_r = self.skeleton.add_bone("eye_r", "head")
        eye_r.local_bind_transform.set_position(0.06, 0.08, 0.12)

        # Set initial pose as bind pose
        self.skeleton.reset_pose()
        for bone in self.skeleton.bones:
            bone.local_transform.position = bone.local_bind_transform.position
            bone.local_transform.rotation = bone.local_bind_transform.rotation
            bone.local_transform.scale_factor = bone.local_bind_transform.scale_factor

        self.skeleton.set_bind_pose()

    def _build_meshes(self) -> None:
        """Build all body part meshes."""
        self._build_torso()
        self._build_head()
        self._build_arms()

    def _build_torso(self) -> None:
        """Build torso meshes."""
        # Hips
        hips_builder = create_capsule(0.12, 0.15, 12, 8, self.BONE_HIPS)
        hips_builder.transform(Mat4.translation(0, 0.9, 0))
        hips_part = AvatarBodyPart("hips", hips_builder.build(), self.BONE_HIPS)
        hips_part.set_color(*self.colors.body)
        self.body_parts.append(hips_part)

        # Spine/Abdomen
        spine_builder = create_capsule(0.11, 0.2, 12, 8, self.BONE_SPINE)
        spine_builder.transform(Mat4.translation(0, 1.1, 0))
        spine_part = AvatarBodyPart("spine", spine_builder.build(), self.BONE_SPINE)
        spine_part.set_color(*self.colors.body)
        self.body_parts.append(spine_part)

        # Chest
        chest_builder = create_capsule(0.14, 0.25, 12, 8, self.BONE_CHEST)
        chest_builder.transform(Mat4.translation(0, 1.35, 0))
        chest_part = AvatarBodyPart("chest", chest_builder.build(), self.BONE_CHEST)
        chest_part.set_color(*self.colors.body_accent)
        self.body_parts.append(chest_part)

        # Neck
        neck_builder = create_cylinder(0.04, 0.1, 10, self.BONE_NECK)
        neck_builder.transform(Mat4.translation(0, 1.55, 0))
        neck_part = AvatarBodyPart("neck", neck_builder.build(), self.BONE_NECK)
        neck_part.set_color(*self.colors.skin)
        self.body_parts.append(neck_part)

    def _build_head(self) -> None:
        """Build head and face meshes."""
        # Main head shape (slightly oval sphere)
        head_builder = create_sphere(0.12, 16, 12, self.BONE_HEAD)
        head_builder.transform(Mat4.scale(1.0, 1.1, 0.95) @ Mat4.translation(0, 1.7, 0))
        head_part = AvatarBodyPart("head", head_builder.build(), self.BONE_HEAD)
        head_part.set_color(*self.colors.skin)
        head_part.set_specular(0.2, 0.15, 0.1, 16.0)
        self.body_parts.append(head_part)

        # Forehead/cranium bump
        cranium_builder = create_sphere(0.1, 14, 10, self.BONE_HEAD)
        cranium_builder.transform(Mat4.scale(1.1, 0.8, 0.9) @ Mat4.translation(0, 1.78, -0.02))
        cranium_part = AvatarBodyPart("cranium", cranium_builder.build(), self.BONE_HEAD)
        cranium_part.set_color(*self.colors.skin)
        self.body_parts.append(cranium_part)

        # Eye sockets (slight indentations modeled as spheres with skin color)
        for x_sign in [-1, 1]:
            socket_builder = create_sphere(0.035, 10, 8, self.BONE_HEAD)
            socket_builder.transform(Mat4.translation(x_sign * 0.045, 1.72, 0.09))
            socket_part = AvatarBodyPart(f"eye_socket_{'l' if x_sign < 0 else 'r'}",
                                         socket_builder.build(), self.BONE_HEAD)
            socket_part.set_color(*self.colors.skin)
            self.body_parts.append(socket_part)

        # Eyeballs
        for x_sign, bone_idx in [(-1, self.BONE_EYE_L), (1, self.BONE_EYE_R)]:
            eye_builder = create_sphere(0.028, 12, 10, bone_idx)
            eye_builder.transform(Mat4.translation(x_sign * 0.045, 1.72, 0.10))
            eye_part = AvatarBodyPart(f"eyeball_{'l' if x_sign < 0 else 'r'}",
                                      eye_builder.build(), bone_idx)
            eye_part.set_color(*self.colors.eye_white)
            eye_part.set_specular(1.0, 1.0, 1.0, 128.0)
            self.body_parts.append(eye_part)

            # Iris
            iris_builder = create_sphere(0.018, 10, 8, bone_idx)
            iris_builder.transform(Mat4.translation(x_sign * 0.045, 1.72, 0.125))
            iris_part = AvatarBodyPart(f"iris_{'l' if x_sign < 0 else 'r'}",
                                       iris_builder.build(), bone_idx)
            iris_part.set_color(*self.colors.eye_iris)
            iris_part.set_specular(0.5, 0.5, 0.5, 64.0)
            self.body_parts.append(iris_part)

            # Pupil
            pupil_builder = create_sphere(0.008, 8, 6, bone_idx)
            pupil_builder.transform(Mat4.translation(x_sign * 0.045, 1.72, 0.14))
            pupil_part = AvatarBodyPart(f"pupil_{'l' if x_sign < 0 else 'r'}",
                                        pupil_builder.build(), bone_idx)
            pupil_part.set_color(*self.colors.eye_pupil)
            self.body_parts.append(pupil_part)

        # Nose
        nose_builder = MeshBuilder()
        # Nose bridge
        bridge = create_sphere(0.015, 8, 6, self.BONE_HEAD)
        bridge.transform(Mat4.scale(0.6, 1.5, 1.0) @ Mat4.translation(0, 1.69, 0.11))
        nose_builder = merge_builders(nose_builder, bridge)
        # Nose tip
        tip = create_sphere(0.018, 8, 6, self.BONE_HEAD)
        tip.transform(Mat4.translation(0, 1.66, 0.13))
        nose_builder = merge_builders(nose_builder, tip)
        nose_part = AvatarBodyPart("nose", nose_builder.build(), self.BONE_HEAD)
        nose_part.set_color(*self.colors.skin)
        self.body_parts.append(nose_part)

        # Mouth area
        mouth_builder = create_capsule(0.025, 0.06, 10, 6, self.BONE_HEAD)
        mouth_builder.transform(Mat4.rotation_z(radians(90)) @ Mat4.translation(0, 1.62, 0.1))
        mouth_part = AvatarBodyPart("mouth", mouth_builder.build(), self.BONE_HEAD)
        mouth_part.set_color(
            self.colors.skin[0] * 0.9,
            self.colors.skin[1] * 0.7,
            self.colors.skin[2] * 0.7
        )
        self.body_parts.append(mouth_part)

        # Jaw/chin
        jaw_builder = create_sphere(0.08, 12, 8, self.BONE_JAW)
        jaw_builder.transform(Mat4.scale(0.9, 0.6, 0.8) @ Mat4.translation(0, 1.58, 0.02))
        jaw_part = AvatarBodyPart("jaw", jaw_builder.build(), self.BONE_JAW)
        jaw_part.set_color(*self.colors.skin)
        self.body_parts.append(jaw_part)

        # Ears
        for x_sign in [-1, 1]:
            ear_builder = create_sphere(0.025, 8, 6, self.BONE_HEAD)
            ear_builder.transform(Mat4.scale(0.4, 1.0, 0.6) @ Mat4.translation(x_sign * 0.12, 1.7, 0))
            ear_part = AvatarBodyPart(f"ear_{'l' if x_sign < 0 else 'r'}",
                                      ear_builder.build(), self.BONE_HEAD)
            ear_part.set_color(*self.colors.skin)
            self.body_parts.append(ear_part)

        # Simple hair (stylized cap)
        hair_builder = create_sphere(0.13, 14, 10, self.BONE_HEAD)
        hair_builder.transform(Mat4.scale(1.05, 0.7, 1.0) @ Mat4.translation(0, 1.8, -0.01))
        hair_part = AvatarBodyPart("hair", hair_builder.build(), self.BONE_HEAD)
        hair_part.set_color(*self.colors.hair)
        hair_part.set_specular(0.1, 0.1, 0.1, 8.0)
        self.body_parts.append(hair_part)

    def _build_arms(self) -> None:
        """Build arm meshes."""
        # Left arm
        self._build_arm("l", -1, self.BONE_SHOULDER_L, self.BONE_UPPER_ARM_L,
                        self.BONE_FOREARM_L, self.BONE_HAND_L)
        # Right arm
        self._build_arm("r", 1, self.BONE_SHOULDER_R, self.BONE_UPPER_ARM_R,
                        self.BONE_FOREARM_R, self.BONE_HAND_R)

    def _build_arm(self, side: str, x_sign: int, shoulder_bone: int, upper_arm_bone: int,
                   forearm_bone: int, hand_bone: int) -> None:
        """Build a single arm."""
        x_offset = x_sign * 0.2

        # Shoulder
        shoulder_builder = create_sphere(0.05, 10, 8, shoulder_bone)
        shoulder_builder.transform(Mat4.translation(x_offset, 1.5, 0))
        shoulder_part = AvatarBodyPart(f"shoulder_{side}", shoulder_builder.build(), shoulder_bone)
        shoulder_part.set_color(*self.colors.body_accent)
        self.body_parts.append(shoulder_part)

        # Upper arm
        upper_arm_builder = create_capsule(0.035, 0.22, 10, 6, upper_arm_bone)
        upper_arm_builder.transform(Mat4.translation(x_offset + x_sign * 0.05, 1.38, 0))
        upper_arm_part = AvatarBodyPart(f"upper_arm_{side}", upper_arm_builder.build(), upper_arm_bone)
        upper_arm_part.set_color(*self.colors.body)
        self.body_parts.append(upper_arm_part)

        # Elbow joint
        elbow_builder = create_sphere(0.03, 8, 6, upper_arm_bone)
        elbow_builder.transform(Mat4.translation(x_offset + x_sign * 0.05, 1.25, 0))
        elbow_part = AvatarBodyPart(f"elbow_{side}", elbow_builder.build(), upper_arm_bone)
        elbow_part.set_color(*self.colors.body_accent)
        self.body_parts.append(elbow_part)

        # Forearm
        forearm_builder = create_capsule(0.03, 0.2, 10, 6, forearm_bone)
        forearm_builder.transform(Mat4.translation(x_offset + x_sign * 0.05, 1.12, 0))
        forearm_part = AvatarBodyPart(f"forearm_{side}", forearm_builder.build(), forearm_bone)
        forearm_part.set_color(*self.colors.skin)
        self.body_parts.append(forearm_part)

        # Wrist joint
        wrist_builder = create_sphere(0.025, 8, 6, forearm_bone)
        wrist_builder.transform(Mat4.translation(x_offset + x_sign * 0.05, 1.0, 0))
        wrist_part = AvatarBodyPart(f"wrist_{side}", wrist_builder.build(), forearm_bone)
        wrist_part.set_color(*self.colors.skin)
        self.body_parts.append(wrist_part)

        # Hand
        hand_builder = create_box(0.06, 0.08, 0.025, hand_bone)
        hand_builder.transform(Mat4.translation(x_offset + x_sign * 0.05, 0.92, 0))
        hand_part = AvatarBodyPart(f"hand_{side}", hand_builder.build(), hand_bone)
        hand_part.set_color(*self.colors.skin)
        self.body_parts.append(hand_part)

        # Simplified fingers (thumb and grouped fingers)
        # Thumb
        thumb_builder = create_capsule(0.012, 0.04, 6, 4, hand_bone)
        thumb_builder.transform(
            Mat4.rotation_z(radians(x_sign * 45)) @
            Mat4.translation(x_offset + x_sign * 0.08, 0.91, 0.01)
        )
        thumb_part = AvatarBodyPart(f"thumb_{side}", thumb_builder.build(), hand_bone)
        thumb_part.set_color(*self.colors.skin)
        self.body_parts.append(thumb_part)

        # Grouped fingers
        for i, finger_x in enumerate([-0.015, 0, 0.015, 0.03]):
            finger_builder = create_capsule(0.008, 0.05, 6, 4, hand_bone)
            finger_builder.transform(
                Mat4.translation(x_offset + x_sign * 0.05 + x_sign * finger_x, 0.86, 0)
            )
            finger_part = AvatarBodyPart(f"finger_{side}_{i}", finger_builder.build(), hand_bone)
            finger_part.set_color(*self.colors.skin)
            self.body_parts.append(finger_part)

    def _setup_animations(self) -> None:
        """Set up the animation controller with default animations."""
        self.animation_controller = AnimationController(self.skeleton)

        # Add built-in animations
        self.animation_controller.add_animation(create_idle_animation())
        self.animation_controller.add_animation(create_speaking_animation())
        self.animation_controller.add_animation(create_wave_animation())

        # Start with idle animation
        self.animation_controller.play("idle", weight=1.0)

        # Set up procedural animator
        self.procedural_animator = ProceduralAnimator(self.skeleton)

    def set_speaking(self, speaking: bool) -> None:
        """Set whether the avatar is speaking."""
        if self.procedural_animator:
            self.procedural_animator.set_speaking(speaking)

        if self.animation_controller:
            if speaking:
                self.animation_controller.set_weight("idle", 0.3)
                self.animation_controller.play("speaking", weight=0.7)
            else:
                self.animation_controller.stop("speaking")
                self.animation_controller.set_weight("idle", 1.0)

    def set_blink(self, amount: float) -> None:
        """Set blink amount (0 = open, 1 = closed)."""
        self.blink_amount = max(0.0, min(1.0, amount))

    def set_look_direction(self, x: float, y: float) -> None:
        """Set where the avatar is looking (-1 to 1 range)."""
        self.look_direction = (
            max(-1.0, min(1.0, x)),
            max(-1.0, min(1.0, y))
        )
        if self.procedural_animator:
            self.procedural_animator.target_look_direction = Vec3(x, y, 0)

    def update(self, delta_time: float) -> None:
        """Update animations."""
        if self.animation_controller:
            self.animation_controller.update(delta_time)

        if self.procedural_animator:
            self.procedural_animator.update(delta_time)
            self.blink_amount = self.procedural_animator.blink_amount

        # Update skeleton after all animation updates
        self.skeleton.update()

    def get_bone_matrices(self) -> List[Mat4]:
        """Get bone matrices for shader upload."""
        return self.skeleton.get_bone_matrices()

    def delete(self) -> None:
        """Clean up GPU resources."""
        for part in self.body_parts:
            part.mesh.delete()
        self.body_parts.clear()
