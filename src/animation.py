"""
Skeletal Animation System for CoFoc 3D Avatar

Provides bones, skeleton, animation clips, and animation blending.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from math3d import Vec3, Quaternion, Mat4, Transform, lerp


@dataclass
class Bone:
    """A single bone in a skeleton."""
    name: str
    index: int
    parent_index: int = -1
    local_bind_transform: Transform = field(default_factory=Transform)
    inverse_bind_matrix: Mat4 = field(default_factory=Mat4.identity)

    # Runtime pose data
    local_transform: Transform = field(default_factory=Transform)
    world_matrix: Mat4 = field(default_factory=Mat4.identity)
    final_matrix: Mat4 = field(default_factory=Mat4.identity)


@dataclass
class Keyframe:
    """A single keyframe in an animation."""
    time: float
    position: Optional[Vec3] = None
    rotation: Optional[Quaternion] = None
    scale: Optional[Vec3] = None


@dataclass
class BoneAnimation:
    """Animation data for a single bone."""
    bone_name: str
    keyframes: List[Keyframe] = field(default_factory=list)

    def sample(self, time: float) -> Tuple[Optional[Vec3], Optional[Quaternion], Optional[Vec3]]:
        """Sample the animation at a given time."""
        if not self.keyframes:
            return None, None, None

        if len(self.keyframes) == 1:
            kf = self.keyframes[0]
            return kf.position, kf.rotation, kf.scale

        # Find surrounding keyframes
        prev_kf = self.keyframes[0]
        next_kf = self.keyframes[-1]

        for i, kf in enumerate(self.keyframes):
            if kf.time <= time:
                prev_kf = kf
            if kf.time >= time and i > 0:
                next_kf = kf
                break

        if prev_kf.time == next_kf.time:
            return prev_kf.position, prev_kf.rotation, prev_kf.scale

        # Interpolation factor
        t = (time - prev_kf.time) / (next_kf.time - prev_kf.time)
        t = max(0.0, min(1.0, t))

        # Interpolate each component
        pos = None
        if prev_kf.position and next_kf.position:
            pos = prev_kf.position.lerp(next_kf.position, t)
        elif prev_kf.position:
            pos = prev_kf.position

        rot = None
        if prev_kf.rotation and next_kf.rotation:
            rot = Quaternion.slerp(prev_kf.rotation, next_kf.rotation, t)
        elif prev_kf.rotation:
            rot = prev_kf.rotation

        scale = None
        if prev_kf.scale and next_kf.scale:
            scale = prev_kf.scale.lerp(next_kf.scale, t)
        elif prev_kf.scale:
            scale = prev_kf.scale

        return pos, rot, scale


@dataclass
class AnimationClip:
    """A complete animation clip."""
    name: str
    duration: float = 1.0
    looping: bool = True
    bone_animations: Dict[str, BoneAnimation] = field(default_factory=dict)

    def add_bone_animation(self, bone_name: str) -> BoneAnimation:
        """Add a new bone animation track."""
        anim = BoneAnimation(bone_name)
        self.bone_animations[bone_name] = anim
        return anim


class Skeleton:
    """Hierarchical bone structure for skeletal animation."""

    def __init__(self):
        self.bones: List[Bone] = []
        self.bone_map: Dict[str, int] = {}
        self.root_bones: List[int] = []

    def add_bone(self, name: str, parent_name: Optional[str] = None) -> Bone:
        """Add a bone to the skeleton."""
        index = len(self.bones)
        parent_index = -1

        if parent_name and parent_name in self.bone_map:
            parent_index = self.bone_map[parent_name]

        bone = Bone(name=name, index=index, parent_index=parent_index)
        self.bones.append(bone)
        self.bone_map[name] = index

        if parent_index == -1:
            self.root_bones.append(index)

        return bone

    def get_bone(self, name: str) -> Optional[Bone]:
        """Get a bone by name."""
        if name in self.bone_map:
            return self.bones[self.bone_map[name]]
        return None

    def set_bind_pose(self) -> None:
        """Calculate and store the inverse bind matrices."""
        self._update_world_matrices()

        for bone in self.bones:
            bone.inverse_bind_matrix = bone.world_matrix.inverse()

    def _update_world_matrices(self) -> None:
        """Update world matrices for all bones."""
        for bone in self.bones:
            local_matrix = bone.local_transform.to_matrix()

            if bone.parent_index >= 0:
                parent = self.bones[bone.parent_index]
                bone.world_matrix = parent.world_matrix @ local_matrix
            else:
                bone.world_matrix = local_matrix

    def update(self) -> None:
        """Update bone matrices after pose changes."""
        self._update_world_matrices()

        for bone in self.bones:
            bone.final_matrix = bone.world_matrix @ bone.inverse_bind_matrix

    def get_bone_matrices(self) -> List[Mat4]:
        """Get final bone matrices for shader upload."""
        return [bone.final_matrix for bone in self.bones]

    def reset_pose(self) -> None:
        """Reset all bones to bind pose."""
        for bone in self.bones:
            bone.local_transform = Transform()
            bone.local_transform.position = bone.local_bind_transform.position
            bone.local_transform.rotation = bone.local_bind_transform.rotation
            bone.local_transform.scale_factor = bone.local_bind_transform.scale_factor


class AnimationState:
    """Manages playback of an animation clip."""

    def __init__(self, clip: AnimationClip):
        self.clip = clip
        self.time = 0.0
        self.speed = 1.0
        self.weight = 1.0
        self.playing = True

    def update(self, delta_time: float) -> None:
        """Update animation time."""
        if not self.playing:
            return

        self.time += delta_time * self.speed

        if self.clip.looping:
            while self.time >= self.clip.duration:
                self.time -= self.clip.duration
            while self.time < 0:
                self.time += self.clip.duration
        else:
            self.time = max(0.0, min(self.time, self.clip.duration))

    def apply_to_skeleton(self, skeleton: Skeleton, blend_weight: float = 1.0) -> None:
        """Apply current animation frame to skeleton."""
        effective_weight = self.weight * blend_weight

        for bone_name, bone_anim in self.clip.bone_animations.items():
            bone = skeleton.get_bone(bone_name)
            if not bone:
                continue

            pos, rot, scale = bone_anim.sample(self.time)

            if pos:
                if effective_weight >= 1.0:
                    bone.local_transform.position = pos
                else:
                    bone.local_transform.position = bone.local_transform.position.lerp(
                        pos, effective_weight)

            if rot:
                if effective_weight >= 1.0:
                    bone.local_transform.rotation = rot
                else:
                    bone.local_transform.rotation = Quaternion.slerp(
                        bone.local_transform.rotation, rot, effective_weight)

            if scale:
                if effective_weight >= 1.0:
                    bone.local_transform.scale_factor = scale
                else:
                    bone.local_transform.scale_factor = bone.local_transform.scale_factor.lerp(
                        scale, effective_weight)


class AnimationController:
    """High-level animation controller with blending support."""

    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton
        self.animations: Dict[str, AnimationClip] = {}
        self.active_states: Dict[str, AnimationState] = {}
        self.blend_weights: Dict[str, float] = {}

    def add_animation(self, clip: AnimationClip) -> None:
        """Register an animation clip."""
        self.animations[clip.name] = clip

    def play(self, name: str, weight: float = 1.0, speed: float = 1.0) -> Optional[AnimationState]:
        """Start playing an animation."""
        if name not in self.animations:
            return None

        state = AnimationState(self.animations[name])
        state.speed = speed
        state.weight = weight
        self.active_states[name] = state
        self.blend_weights[name] = weight
        return state

    def stop(self, name: str) -> None:
        """Stop an animation."""
        if name in self.active_states:
            del self.active_states[name]
        if name in self.blend_weights:
            del self.blend_weights[name]

    def set_weight(self, name: str, weight: float) -> None:
        """Set the blend weight of an animation."""
        if name in self.active_states:
            self.active_states[name].weight = weight
            self.blend_weights[name] = weight

    def fade_to(self, name: str, duration: float, target_weight: float = 1.0) -> None:
        """Fade to an animation over time (simplified - immediate for now)."""
        # For a full implementation, this would use a blend tree
        # For now, just set weight directly
        self.set_weight(name, target_weight)

    def update(self, delta_time: float) -> None:
        """Update all animations and apply to skeleton."""
        self.skeleton.reset_pose()

        # Update all animation states
        for state in self.active_states.values():
            state.update(delta_time)

        # Apply animations with blending
        total_weight = sum(self.blend_weights.values())
        if total_weight <= 0:
            total_weight = 1.0

        for name, state in self.active_states.items():
            normalized_weight = self.blend_weights.get(name, 0) / total_weight
            state.apply_to_skeleton(self.skeleton, normalized_weight)

        self.skeleton.update()


def create_idle_animation() -> AnimationClip:
    """Create a simple idle breathing animation."""
    clip = AnimationClip(name="idle", duration=4.0, looping=True)

    # Spine breathing
    spine_anim = clip.add_bone_animation("spine")
    spine_anim.keyframes = [
        Keyframe(0.0, scale=Vec3(1.0, 1.0, 1.0)),
        Keyframe(2.0, scale=Vec3(1.0, 1.02, 1.0)),
        Keyframe(4.0, scale=Vec3(1.0, 1.0, 1.0)),
    ]

    # Head subtle movement
    head_anim = clip.add_bone_animation("head")
    head_anim.keyframes = [
        Keyframe(0.0, rotation=Quaternion.identity()),
        Keyframe(1.0, rotation=Quaternion.from_euler(0.02, 0.01, 0)),
        Keyframe(2.0, rotation=Quaternion.from_euler(0, 0, 0)),
        Keyframe(3.0, rotation=Quaternion.from_euler(-0.02, -0.01, 0)),
        Keyframe(4.0, rotation=Quaternion.identity()),
    ]

    # Shoulder subtle movement
    for side, sign in [("shoulder_l", -1), ("shoulder_r", 1)]:
        shoulder_anim = clip.add_bone_animation(side)
        shoulder_anim.keyframes = [
            Keyframe(0.0, rotation=Quaternion.identity()),
            Keyframe(2.0, rotation=Quaternion.from_euler(0, 0, sign * 0.02)),
            Keyframe(4.0, rotation=Quaternion.identity()),
        ]

    return clip


def create_speaking_animation() -> AnimationClip:
    """Create a speaking animation with head and arm movement."""
    clip = AnimationClip(name="speaking", duration=1.0, looping=True)

    # Head nods while speaking
    head_anim = clip.add_bone_animation("head")
    head_anim.keyframes = [
        Keyframe(0.0, rotation=Quaternion.identity()),
        Keyframe(0.25, rotation=Quaternion.from_euler(0.05, 0.03, 0)),
        Keyframe(0.5, rotation=Quaternion.from_euler(-0.03, -0.02, 0)),
        Keyframe(0.75, rotation=Quaternion.from_euler(0.02, 0.01, 0)),
        Keyframe(1.0, rotation=Quaternion.identity()),
    ]

    # Right arm gestures
    arm_anim = clip.add_bone_animation("upper_arm_r")
    arm_anim.keyframes = [
        Keyframe(0.0, rotation=Quaternion.from_euler(0, 0, -0.3)),
        Keyframe(0.3, rotation=Quaternion.from_euler(0.1, 0, -0.5)),
        Keyframe(0.6, rotation=Quaternion.from_euler(-0.05, 0, -0.35)),
        Keyframe(1.0, rotation=Quaternion.from_euler(0, 0, -0.3)),
    ]

    forearm_anim = clip.add_bone_animation("forearm_r")
    forearm_anim.keyframes = [
        Keyframe(0.0, rotation=Quaternion.from_euler(0, 0, 0.5)),
        Keyframe(0.3, rotation=Quaternion.from_euler(0, 0, 0.7)),
        Keyframe(0.6, rotation=Quaternion.from_euler(0, 0, 0.4)),
        Keyframe(1.0, rotation=Quaternion.from_euler(0, 0, 0.5)),
    ]

    return clip


def create_wave_animation() -> AnimationClip:
    """Create a waving animation."""
    clip = AnimationClip(name="wave", duration=2.0, looping=True)

    # Raise right arm
    arm_anim = clip.add_bone_animation("upper_arm_r")
    arm_anim.keyframes = [
        Keyframe(0.0, rotation=Quaternion.from_euler(0, 0, -1.2)),
        Keyframe(2.0, rotation=Quaternion.from_euler(0, 0, -1.2)),
    ]

    # Forearm waves
    forearm_anim = clip.add_bone_animation("forearm_r")
    forearm_anim.keyframes = [
        Keyframe(0.0, rotation=Quaternion.from_euler(0.3, 0.3, 0.5)),
        Keyframe(0.25, rotation=Quaternion.from_euler(-0.3, -0.3, 0.5)),
        Keyframe(0.5, rotation=Quaternion.from_euler(0.3, 0.3, 0.5)),
        Keyframe(0.75, rotation=Quaternion.from_euler(-0.3, -0.3, 0.5)),
        Keyframe(1.0, rotation=Quaternion.from_euler(0.3, 0.3, 0.5)),
        Keyframe(1.25, rotation=Quaternion.from_euler(-0.3, -0.3, 0.5)),
        Keyframe(1.5, rotation=Quaternion.from_euler(0.3, 0.3, 0.5)),
        Keyframe(1.75, rotation=Quaternion.from_euler(-0.3, -0.3, 0.5)),
        Keyframe(2.0, rotation=Quaternion.from_euler(0.3, 0.3, 0.5)),
    ]

    # Hand rotates
    hand_anim = clip.add_bone_animation("hand_r")
    hand_anim.keyframes = [
        Keyframe(0.0, rotation=Quaternion.from_euler(0, 0, 0.3)),
        Keyframe(0.25, rotation=Quaternion.from_euler(0, 0, -0.3)),
        Keyframe(0.5, rotation=Quaternion.from_euler(0, 0, 0.3)),
        Keyframe(0.75, rotation=Quaternion.from_euler(0, 0, -0.3)),
        Keyframe(1.0, rotation=Quaternion.from_euler(0, 0, 0.3)),
        Keyframe(1.25, rotation=Quaternion.from_euler(0, 0, -0.3)),
        Keyframe(1.5, rotation=Quaternion.from_euler(0, 0, 0.3)),
        Keyframe(1.75, rotation=Quaternion.from_euler(0, 0, -0.3)),
        Keyframe(2.0, rotation=Quaternion.from_euler(0, 0, 0.3)),
    ]

    return clip


class ProceduralAnimator:
    """Procedural animation layer for dynamic effects with phoneme-based lip sync."""

    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton
        self.time = 0.0

        # Mouth animation state
        self.mouth_openness = 0.0
        self.target_mouth_openness = 0.0
        self.is_speaking = False

        # Blink state
        self.blink_timer = 0.0
        self.blink_duration = 0.15
        self.blink_interval = 3.0
        self.is_blinking = False
        self.blink_amount = 0.0

        # Look direction
        self.look_direction = Vec3(0, 0, 0)
        self.target_look_direction = Vec3(0, 0, 0)

        # Breathing
        self.breathing_phase = 0.0

        # Lip sync engine reference (lazy loaded)
        self._lip_sync = None

    def _get_lip_sync(self):
        """Get the lip sync engine (lazy load to avoid circular imports)."""
        if self._lip_sync is None:
            try:
                from lipsync import get_lip_sync_engine
                self._lip_sync = get_lip_sync_engine()
            except ImportError:
                pass
        return self._lip_sync

    def set_speaking(self, speaking: bool) -> None:
        """Set whether the avatar is speaking."""
        self.is_speaking = speaking
        if not speaking:
            self.target_mouth_openness = 0.0

    def update(self, delta_time: float) -> None:
        """Update procedural animations with phoneme-based lip sync."""
        self.time += delta_time

        # Update mouth using lip sync engine if available
        lip_sync = self._get_lip_sync()
        if lip_sync and lip_sync.is_playing:
            # Get viseme-based mouth openness from phoneme timeline
            self.mouth_openness = lip_sync.get_simple_mouth_open()
        elif self.is_speaking:
            # Fallback: random mouth movement when speaking
            import random
            self.target_mouth_openness = random.uniform(0.3, 1.0)
            self.mouth_openness += (self.target_mouth_openness - self.mouth_openness) * 10 * delta_time
        else:
            self.mouth_openness += (0.0 - self.mouth_openness) * 10 * delta_time

        # Update blinking
        self.blink_timer += delta_time
        if not self.is_blinking and self.blink_timer >= self.blink_interval:
            self.is_blinking = True
            self.blink_timer = 0.0
            import random
            self.blink_interval = random.uniform(2.0, 5.0)

        if self.is_blinking:
            self.blink_amount = min(1.0, self.blink_amount + delta_time / (self.blink_duration * 0.5))
            if self.blink_amount >= 1.0:
                self.blink_amount = max(0.0, self.blink_amount - delta_time / (self.blink_duration * 0.5))
                if self.blink_amount <= 0.0:
                    self.is_blinking = False
                    self.blink_amount = 0.0

        # Update look direction
        self.look_direction = self.look_direction.lerp(self.target_look_direction, 5 * delta_time)

        # Update breathing
        self.breathing_phase += delta_time * 0.5
        breath = math.sin(self.breathing_phase * 2 * math.pi) * 0.5 + 0.5

        # Apply to skeleton
        jaw = self.skeleton.get_bone("jaw")
        if jaw:
            jaw.local_transform.rotation = Quaternion.from_euler(
                self.mouth_openness * 0.3, 0, 0
            )

        # Apply look direction to head
        head = self.skeleton.get_bone("head")
        if head:
            current_rot = head.local_transform.rotation
            look_rot = Quaternion.from_euler(
                self.look_direction.y * 0.2,
                self.look_direction.x * 0.3,
                0
            )
            head.local_transform.rotation = current_rot * look_rot

        # Apply breathing to spine
        spine = self.skeleton.get_bone("spine")
        if spine:
            spine.local_transform.scale_factor = Vec3(
                1.0,
                1.0 + breath * 0.01,
                1.0 + breath * 0.02
            )
