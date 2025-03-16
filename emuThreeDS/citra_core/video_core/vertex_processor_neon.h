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
#include "video_core/pica_types.h"
#include "video_core/shader_processing_neon.h"

namespace Pica {
namespace VertexProcessor {
namespace NEON {

/**
 * Optimized vertex transformation using NEON intrinsics
 * @param dst_array Destination array for transformed vertices
 * @param src_array Source array of vertices to transform
 * @param num_vertices Number of vertices to process
 * @param transform Transformation matrix
 */
inline void TransformVertices(Common::Vec4<float>* dst_array, const Common::Vec4<float>* src_array, 
                             size_t num_vertices, const Pica::Matrix4x4& transform) {
    // Process vertices in batches of 4 for better NEON utilization
    size_t i = 0;
    for (; i + 3 < num_vertices; i += 4) {
        // Load 4 vertices into NEON registers
        float32x4_t vx0 = {src_array[i].x, src_array[i+1].x, src_array[i+2].x, src_array[i+3].x};
        float32x4_t vy0 = {src_array[i].y, src_array[i+1].y, src_array[i+2].y, src_array[i+3].y};
        float32x4_t vz0 = {src_array[i].z, src_array[i+1].z, src_array[i+2].z, src_array[i+3].z};
        float32x4_t vw0 = {src_array[i].w, src_array[i+1].w, src_array[i+2].w, src_array[i+3].w};
        
        // Load transformation matrix rows
        float32x4_t row0 = vld1q_f32(&transform.r[0].x);
        float32x4_t row1 = vld1q_f32(&transform.r[1].x);
        float32x4_t row2 = vld1q_f32(&transform.r[2].x);
        float32x4_t row3 = vld1q_f32(&transform.r[3].x);
        
        // Transform X coordinates
        float32x4_t outX;
        outX = vmulq_f32(vdupq_n_f32(row0[0]), vx0);
        outX = vmlaq_f32(outX, vdupq_n_f32(row0[1]), vy0);
        outX = vmlaq_f32(outX, vdupq_n_f32(row0[2]), vz0);
        outX = vmlaq_f32(outX, vdupq_n_f32(row0[3]), vw0);
        
        // Transform Y coordinates
        float32x4_t outY;
        outY = vmulq_f32(vdupq_n_f32(row1[0]), vx0);
        outY = vmlaq_f32(outY, vdupq_n_f32(row1[1]), vy0);
        outY = vmlaq_f32(outY, vdupq_n_f32(row1[2]), vz0);
        outY = vmlaq_f32(outY, vdupq_n_f32(row1[3]), vw0);
        
        // Transform Z coordinates
        float32x4_t outZ;
        outZ = vmulq_f32(vdupq_n_f32(row2[0]), vx0);
        outZ = vmlaq_f32(outZ, vdupq_n_f32(row2[1]), vy0);
        outZ = vmlaq_f32(outZ, vdupq_n_f32(row2[2]), vz0);
        outZ = vmlaq_f32(outZ, vdupq_n_f32(row2[3]), vw0);
        
        // Transform W coordinates
        float32x4_t outW;
        outW = vmulq_f32(vdupq_n_f32(row3[0]), vx0);
        outW = vmlaq_f32(outW, vdupq_n_f32(row3[1]), vy0);
        outW = vmlaq_f32(outW, vdupq_n_f32(row3[2]), vz0);
        outW = vmlaq_f32(outW, vdupq_n_f32(row3[3]), vw0);
        
        // Store results
        dst_array[i].x = vgetq_lane_f32(outX, 0);
        dst_array[i].y = vgetq_lane_f32(outY, 0);
        dst_array[i].z = vgetq_lane_f32(outZ, 0);
        dst_array[i].w = vgetq_lane_f32(outW, 0);
        
        dst_array[i+1].x = vgetq_lane_f32(outX, 1);
        dst_array[i+1].y = vgetq_lane_f32(outY, 1);
        dst_array[i+1].z = vgetq_lane_f32(outZ, 1);
        dst_array[i+1].w = vgetq_lane_f32(outW, 1);
        
        dst_array[i+2].x = vgetq_lane_f32(outX, 2);
        dst_array[i+2].y = vgetq_lane_f32(outY, 2);
        dst_array[i+2].z = vgetq_lane_f32(outZ, 2);
        dst_array[i+2].w = vgetq_lane_f32(outW, 2);
        
        dst_array[i+3].x = vgetq_lane_f32(outX, 3);
        dst_array[i+3].y = vgetq_lane_f32(outY, 3);
        dst_array[i+3].z = vgetq_lane_f32(outZ, 3);
        dst_array[i+3].w = vgetq_lane_f32(outW, 3);
    }
    
    // Handle remaining vertices
    for (; i < num_vertices; i++) {
        dst_array[i] = Common::NEON::MultiplyMatrixVector(transform, src_array[i]);
    }
}

/**
 * Optimized vertex lighting calculation using NEON intrinsics
 * @param dst_array Destination array for lit vertices
 * @param src_array Source array of vertices to light
 * @param normal_array Array of vertex normals
 * @param num_vertices Number of vertices to process
 * @param light_direction Light direction vector
 * @param light_color Light color
 * @param ambient_color Ambient color
 */
inline void LightVertices(Common::Vec4<float>* dst_array, const Common::Vec4<float>* src_array,
                         const Common::Vec3<float>* normal_array, size_t num_vertices,
                         const Common::Vec3<float>& light_direction,
                         const Common::Vec4<float>& light_color,
                         const Common::Vec4<float>& ambient_color) {
    // Load light direction into NEON register
    float32x4_t light_dir = vdupq_n_f32(0.0f);
    light_dir = vsetq_lane_f32(light_direction.x, light_dir, 0);
    light_dir = vsetq_lane_f32(light_direction.y, light_dir, 1);
    light_dir = vsetq_lane_f32(light_direction.z, light_dir, 2);
    
    // Load light color into NEON register
    float32x4_t light = vld1q_f32(&light_color.x);
    
    // Load ambient color into NEON register
    float32x4_t ambient = vld1q_f32(&ambient_color.x);
    
    // Process vertices
    for (size_t i = 0; i < num_vertices; i++) {
        // Load normal into NEON register
        float32x4_t normal = vdupq_n_f32(0.0f);
        normal = vsetq_lane_f32(normal_array[i].x, normal, 0);
        normal = vsetq_lane_f32(normal_array[i].y, normal, 1);
        normal = vsetq_lane_f32(normal_array[i].z, normal, 2);
        
        // Calculate diffuse factor (dot product of normal and light direction)
        float diffuse_factor = Common::NEON::Dot3(normal_array[i], light_direction);
        diffuse_factor = std::max(0.0f, diffuse_factor);
        
        // Load source color into NEON register
        float32x4_t src_color = vld1q_f32(&src_array[i].x);
        
        // Calculate ambient component
        float32x4_t result = vmulq_f32(src_color, ambient);
        
        // Calculate diffuse component and add to result
        float32x4_t diffuse = vmulq_f32(src_color, light);
        diffuse = vmulq_n_f32(diffuse, diffuse_factor);
        result = vaddq_f32(result, diffuse);
        
        // Clamp result to [0, 1]
        result = vminq_f32(result, vdupq_n_f32(1.0f));
        result = vmaxq_f32(result, vdupq_n_f32(0.0f));
        
        // Store result
        vst1q_f32(&dst_array[i].x, result);
    }
}

} // namespace NEON
} // namespace VertexProcessor
} // namespace Pica

#endif // defined(__ARM_NEON) || defined(__aarch64__)
