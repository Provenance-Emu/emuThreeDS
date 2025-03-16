// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

// Only use ARM NEON intrinsics on ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#include "common/vector_math.h"
#include "common/vector_math_neon.h"
#include "common/matrix_math_neon.h"
#include "video_core/pica_types.h"
#include "video_core/neon_optimizations.h"

namespace Pica {
namespace Shader {
namespace NEON {

/**
 * Optimized vertex shader attribute transformation using NEON intrinsics
 * @param output Output transformed attribute
 * @param input Input attribute
 * @param matrix Transformation matrix
 */
inline void TransformVec4(Common::Vec4<float>& output, const Common::Vec4<float>& input, 
                         const Pica::Matrix4x4& matrix) {
    output = Common::NEON::MultiplyMatrixVector(matrix, input);
}

/**
 * Optimized normal transformation using NEON intrinsics
 * @param output Output normal
 * @param input Input normal
 * @param normal_matrix Normal matrix (inverse transpose of model-view matrix)
 */
inline void TransformNormal(Common::Vec3<float>& output, const Common::Vec3<float>& input,
                           const Pica::Matrix3x3& normal_matrix) {
    // Use the optimized matrix-vector multiplication from matrix_math_neon.h
    output = Common::NEON::MultiplyMatrixVector(normal_matrix, input);
    
    // Normalize the result
    output = Common::NEON::Normalize(output);
}

/**
 * Optimized batch vertex processing using NEON intrinsics
 * @param output_positions Output vertex positions
 * @param input_positions Input vertex positions
 * @param count Number of vertices to process
 * @param mvp_matrix Model-view-projection matrix
 */
inline void ProcessVerticesBatch(Common::Vec4<float>* output_positions, 
                                const Common::Vec4<float>* input_positions,
                                size_t count, const Pica::Matrix4x4& mvp_matrix) {
    for (size_t i = 0; i < count; i++) {
        TransformVec4(output_positions[i], input_positions[i], mvp_matrix);
    }
}

/**
 * Optimized lighting calculation using NEON intrinsics
 * @param output Output color
 * @param normal Surface normal
 * @param light_direction Light direction
 * @param material_color Material color
 * @param light_color Light color
 * @param ambient_color Ambient color
 */
inline void CalculateLighting(Common::Vec4<float>& output, const Common::Vec3<float>& normal,
                             const Common::Vec3<float>& light_direction,
                             const Common::Vec4<float>& material_color,
                             const Common::Vec4<float>& light_color,
                             const Common::Vec4<float>& ambient_color) {
    // Calculate diffuse factor (dot product of normal and light direction)
    float diffuse_factor = Common::NEON::Dot3(normal, light_direction);
    diffuse_factor = std::max(0.0f, diffuse_factor);
    
    // Load material color into NEON register
    float32x4_t material = vld1q_f32(&material_color.x);
    
    // Load light color into NEON register
    float32x4_t light = vld1q_f32(&light_color.x);
    
    // Load ambient color into NEON register
    float32x4_t ambient = vld1q_f32(&ambient_color.x);
    
    // Calculate ambient component
    float32x4_t result = vmulq_f32(material, ambient);
    
    // Calculate diffuse component and add to result
    float32x4_t diffuse = vmulq_f32(material, light);
    diffuse = vmulq_n_f32(diffuse, diffuse_factor);
    result = vaddq_f32(result, diffuse);
    
    // Clamp result to [0, 1]
    result = vminq_f32(result, vdupq_n_f32(1.0f));
    result = vmaxq_f32(result, vdupq_n_f32(0.0f));
    
    // Store result
    vst1q_f32(&output.x, result);
}

/**
 * Optimized attribute interpolation using NEON intrinsics
 * @param output Output interpolated attribute
 * @param attr0 First attribute
 * @param attr1 Second attribute
 * @param attr2 Third attribute
 * @param weights Barycentric weights
 */
inline void InterpolateAttribute(Common::Vec4<float>& output,
                                const Common::Vec4<float>& attr0,
                                const Common::Vec4<float>& attr1,
                                const Common::Vec4<float>& attr2,
                                const Common::Vec3<float>& weights) {
    // Load attributes into NEON registers
    float32x4_t a0 = vld1q_f32(&attr0.x);
    float32x4_t a1 = vld1q_f32(&attr1.x);
    float32x4_t a2 = vld1q_f32(&attr2.x);
    
    // Multiply each attribute by its weight
    float32x4_t w0 = vdupq_n_f32(weights.x);
    float32x4_t w1 = vdupq_n_f32(weights.y);
    float32x4_t w2 = vdupq_n_f32(weights.z);
    
    float32x4_t result = vmulq_f32(a0, w0);
    result = vmlaq_f32(result, a1, w1);
    result = vmlaq_f32(result, a2, w2);
    
    // Store result
    vst1q_f32(&output.x, result);
}

} // namespace NEON
} // namespace Shader
} // namespace Pica

#endif // defined(__ARM_NEON) || defined(__aarch64__)
