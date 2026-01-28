"""
OpenGL Shader System for CoFoc 3D Avatar

Provides shader compilation, linking, and uniform management.
"""

from OpenGL.GL import *
import numpy as np
from typing import Dict, Optional
from math3d import Mat4, Vec3


class ShaderError(Exception):
    """Exception raised for shader compilation/linking errors."""
    pass


class Shader:
    """OpenGL Shader Program wrapper."""

    def __init__(self):
        self.program_id = 0
        self._uniform_cache: Dict[str, int] = {}

    def compile(self, vertex_source: str, fragment_source: str) -> None:
        """Compile and link vertex and fragment shaders."""
        vertex_shader = self._compile_shader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = self._compile_shader(fragment_source, GL_FRAGMENT_SHADER)

        self.program_id = glCreateProgram()
        glAttachShader(self.program_id, vertex_shader)
        glAttachShader(self.program_id, fragment_shader)
        glLinkProgram(self.program_id)

        if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
            error = glGetProgramInfoLog(self.program_id).decode()
            raise ShaderError(f"Shader linking failed:\n{error}")

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

    def _compile_shader(self, source: str, shader_type: int) -> int:
        """Compile a single shader."""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)

        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            error = glGetShaderInfoLog(shader).decode()
            shader_type_name = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
            raise ShaderError(f"{shader_type_name} shader compilation failed:\n{error}")

        return shader

    def use(self) -> None:
        """Activate this shader program."""
        glUseProgram(self.program_id)

    def _get_uniform_location(self, name: str) -> int:
        """Get cached uniform location."""
        if name not in self._uniform_cache:
            location = glGetUniformLocation(self.program_id, name)
            self._uniform_cache[name] = location
        return self._uniform_cache[name]

    def set_uniform_mat4(self, name: str, matrix: Mat4) -> None:
        """Set a mat4 uniform."""
        location = self._get_uniform_location(name)
        if location >= 0:
            glUniformMatrix4fv(location, 1, GL_TRUE, matrix.data)

    def set_uniform_mat4_array(self, name: str, matrices: list) -> None:
        """Set an array of mat4 uniforms."""
        location = self._get_uniform_location(name)
        if location >= 0 and matrices:
            data = np.array([m.data for m in matrices], dtype=np.float32)
            glUniformMatrix4fv(location, len(matrices), GL_TRUE, data)

    def set_uniform_vec3(self, name: str, v: Vec3) -> None:
        """Set a vec3 uniform."""
        location = self._get_uniform_location(name)
        if location >= 0:
            glUniform3f(location, v.x, v.y, v.z)

    def set_uniform_vec3_values(self, name: str, x: float, y: float, z: float) -> None:
        """Set a vec3 uniform from values."""
        location = self._get_uniform_location(name)
        if location >= 0:
            glUniform3f(location, x, y, z)

    def set_uniform_float(self, name: str, value: float) -> None:
        """Set a float uniform."""
        location = self._get_uniform_location(name)
        if location >= 0:
            glUniform1f(location, value)

    def set_uniform_int(self, name: str, value: int) -> None:
        """Set an int uniform."""
        location = self._get_uniform_location(name)
        if location >= 0:
            glUniform1i(location, value)

    def delete(self) -> None:
        """Delete the shader program."""
        if self.program_id:
            glDeleteProgram(self.program_id)
            self.program_id = 0


# Shader source code for avatar rendering with skeletal animation

AVATAR_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aBoneWeights;
layout(location = 3) in ivec4 aBoneIndices;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uBoneMatrices[64];
uniform int uUseSkinning;

out vec3 vWorldPos;
out vec3 vNormal;

void main() {
    vec4 position = vec4(aPosition, 1.0);
    vec4 normal = vec4(aNormal, 0.0);

    if (uUseSkinning == 1) {
        mat4 skinMatrix =
            uBoneMatrices[aBoneIndices.x] * aBoneWeights.x +
            uBoneMatrices[aBoneIndices.y] * aBoneWeights.y +
            uBoneMatrices[aBoneIndices.z] * aBoneWeights.z +
            uBoneMatrices[aBoneIndices.w] * aBoneWeights.w;

        position = skinMatrix * position;
        normal = skinMatrix * normal;
    }

    vec4 worldPos = uModel * position;
    vWorldPos = worldPos.xyz;
    vNormal = normalize(mat3(transpose(inverse(uModel))) * normal.xyz);

    gl_Position = uProjection * uView * worldPos;
}
"""

AVATAR_FRAGMENT_SHADER = """
#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;

uniform vec3 uCameraPos;
uniform vec3 uLightPos;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uDiffuseColor;
uniform vec3 uSpecularColor;
uniform float uShininess;
uniform float uEmission;

out vec4 fragColor;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightPos - vWorldPos);
    vec3 viewDir = normalize(uCameraPos - vWorldPos);
    vec3 reflectDir = reflect(-lightDir, normal);

    // Ambient
    vec3 ambient = uAmbientColor * uDiffuseColor;

    // Diffuse (half-lambert for softer shading)
    float diff = dot(normal, lightDir);
    diff = diff * 0.5 + 0.5;
    diff = diff * diff;
    vec3 diffuse = uLightColor * uDiffuseColor * diff;

    // Specular (Blinn-Phong)
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), uShininess);
    vec3 specular = uLightColor * uSpecularColor * spec;

    // Rim lighting for stylized look
    float rim = 1.0 - max(dot(viewDir, normal), 0.0);
    rim = smoothstep(0.6, 1.0, rim);
    vec3 rimColor = rim * uLightColor * 0.3;

    // Emission
    vec3 emission = uDiffuseColor * uEmission;

    vec3 result = ambient + diffuse + specular + rimColor + emission;

    // Tone mapping
    result = result / (result + vec3(1.0));

    // Gamma correction
    result = pow(result, vec3(1.0/2.2));

    fragColor = vec4(result, 1.0);
}
"""

# Simple solid color shader for debugging
SOLID_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 aPosition;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

void main() {
    gl_Position = uProjection * uView * uModel * vec4(aPosition, 1.0);
}
"""

SOLID_FRAGMENT_SHADER = """
#version 330 core

uniform vec3 uColor;

out vec4 fragColor;

void main() {
    fragColor = vec4(uColor, 1.0);
}
"""

# Eye shader with special effects
EYE_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;

void main() {
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(mat3(transpose(inverse(uModel))) * aNormal);
    vTexCoord = aTexCoord;

    gl_Position = uProjection * uView * worldPos;
}
"""

EYE_FRAGMENT_SHADER = """
#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;

uniform vec3 uCameraPos;
uniform vec3 uLightPos;
uniform vec3 uEyeColor;
uniform vec3 uPupilColor;
uniform float uPupilSize;
uniform float uBlinkAmount;
uniform vec2 uLookDirection;

out vec4 fragColor;

void main() {
    vec2 uv = vTexCoord * 2.0 - 1.0;

    // Apply blink by squishing UV
    float blinkMask = 1.0 - uBlinkAmount;
    if (abs(uv.y) > blinkMask) {
        discard;
    }

    // Offset for look direction
    vec2 pupilCenter = uLookDirection * 0.2;

    // Distance from pupil center
    float dist = length(uv - pupilCenter);

    // Pupil
    float pupilRadius = uPupilSize;
    float pupil = smoothstep(pupilRadius, pupilRadius - 0.05, dist);

    // Iris
    float irisRadius = 0.7;
    float iris = smoothstep(irisRadius, irisRadius - 0.1, dist);

    // Eye white highlight
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(uCameraPos - vWorldPos);
    float fresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), 3.0);

    // Combine colors
    vec3 color = mix(vec3(1.0), uEyeColor, iris);
    color = mix(color, uPupilColor, pupil);

    // Add highlight
    vec2 highlightPos = vec2(-0.3, 0.3);
    float highlight = smoothstep(0.15, 0.0, length(uv - highlightPos));
    color += highlight * 0.5;

    // Add rim
    color += fresnel * 0.2;

    fragColor = vec4(color, 1.0);
}
"""


def create_avatar_shader() -> Shader:
    """Create the main avatar shader with skeletal animation support."""
    shader = Shader()
    shader.compile(AVATAR_VERTEX_SHADER, AVATAR_FRAGMENT_SHADER)
    return shader


def create_solid_shader() -> Shader:
    """Create a simple solid color shader."""
    shader = Shader()
    shader.compile(SOLID_VERTEX_SHADER, SOLID_FRAGMENT_SHADER)
    return shader


def create_eye_shader() -> Shader:
    """Create the eye shader with special effects."""
    shader = Shader()
    shader.compile(EYE_VERTEX_SHADER, EYE_FRAGMENT_SHADER)
    return shader
