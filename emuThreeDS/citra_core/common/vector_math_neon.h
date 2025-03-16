// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/vector_math.h"

// Only use ARM NEON intrinsics on ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

namespace Common {

/**
 * ARM NEON optimized vector math operations
 * These functions provide SIMD-accelerated implementations of common vector operations
 * used throughout the emulator, particularly in graphics and physics calculations.
 */
namespace NEON {

/**
 * Performs a fast dot product between two 3D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return Dot product result
 */
inline float Dot3(const Vec3f& vec1, const Vec3f& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    // Set individual components
    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    
    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    
    // Multiply vectors component-wise
    float32x4_t mul = vmulq_f32(v1, v2);
    
    // Sum components horizontally
    float32x2_t sum = vpadd_f32(vget_low_f32(mul), vget_high_f32(mul));
    sum = vpadd_f32(sum, sum);
    
    // Extract result
    return vget_lane_f32(sum, 0);
}

/**
 * Computes the cross product of two 3D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return Cross product result
 */
inline Vec3f Cross3(const Vec3f& vec1, const Vec3f& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    
    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    
    // Compute cross product components using shuffled multiplications
    // result.x = v1.y * v2.z - v1.z * v2.y
    // result.y = v1.z * v2.x - v1.x * v2.z
    // result.z = v1.x * v2.y - v1.y * v2.x
    
    // Shuffle v1: (y,z,x)
    float32x4_t v1_yzx = vextq_f32(v1, v1, 1);
    
    // Shuffle v2: (z,x,y)
    float32x4_t v2_zxy = vextq_f32(v2, v2, 2);
    
    // Shuffle v1: (z,x,y)
    float32x4_t v1_zxy = vextq_f32(v1, v1, 2);
    
    // Shuffle v2: (y,z,x)
    float32x4_t v2_yzx = vextq_f32(v2, v2, 1);
    
    // Compute v1_yzx * v2_zxy - v1_zxy * v2_yzx
    float32x4_t mul1 = vmulq_f32(v1_yzx, v2_zxy);
    float32x4_t mul2 = vmulq_f32(v1_zxy, v2_yzx);
    float32x4_t result = vsubq_f32(mul1, mul2);
    
    // Extract results
    Vec3f cross_product;
    cross_product.x = vgetq_lane_f32(result, 0);
    cross_product.y = vgetq_lane_f32(result, 1);
    cross_product.z = vgetq_lane_f32(result, 2);
    
    return cross_product;
}

/**
 * Computes the length (magnitude) of a 3D vector using NEON intrinsics
 * @param vec Vector to compute length for
 * @return Vector length
 */
inline float Length3(const Vec3f& vec) {
    // Load vector components into NEON register
    float32x4_t v = vdupq_n_f32(0.0f);
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);
    
    // Square each component
    float32x4_t v_squared = vmulq_f32(v, v);
    
    // Sum components horizontally
    float32x2_t sum = vpadd_f32(vget_low_f32(v_squared), vget_high_f32(v_squared));
    sum = vpadd_f32(sum, sum);
    
    // Take square root of sum
    float32x2_t length = vsqrt_f32(sum);
    
    // Extract result
    return vget_lane_f32(length, 0);
}

/**
 * Normalizes a 3D vector using NEON intrinsics
 * @param vec Vector to normalize
 * @return Normalized vector
 */
inline Vec3f Normalize3(const Vec3f& vec) {
    // Load vector components into NEON register
    float32x4_t v = vdupq_n_f32(0.0f);
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);
    
    // Square each component
    float32x4_t v_squared = vmulq_f32(v, v);
    
    // Sum components horizontally
    float32x2_t sum = vpadd_f32(vget_low_f32(v_squared), vget_high_f32(v_squared));
    sum = vpadd_f32(sum, sum);
    
    // Take square root of sum to get length
    float32x2_t length = vsqrt_f32(sum);
    
    // Broadcast length to all lanes
    float32x4_t length_splat = vdupq_n_f32(vget_lane_f32(length, 0));
    
    // Divide vector by length to normalize
    float32x4_t normalized = vdivq_f32(v, length_splat);
    
    // Extract results
    Vec3f result;
    result.x = vgetq_lane_f32(normalized, 0);
    result.y = vgetq_lane_f32(normalized, 1);
    result.z = vgetq_lane_f32(normalized, 2);
    
    return result;
}

/**
 * Performs fast vector addition of two 3D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return Sum vector
 */
inline Vec3f Add3(const Vec3f& vec1, const Vec3f& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    
    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    
    // Add vectors
    float32x4_t result = vaddq_f32(v1, v2);
    
    // Extract results
    Vec3f sum;
    sum.x = vgetq_lane_f32(result, 0);
    sum.y = vgetq_lane_f32(result, 1);
    sum.z = vgetq_lane_f32(result, 2);
    
    return sum;
}

/**
 * Performs fast vector subtraction of two 3D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return Difference vector
 */
inline Vec3f Subtract3(const Vec3f& vec1, const Vec3f& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    
    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    
    // Subtract vectors
    float32x4_t result = vsubq_f32(v1, v2);
    
    // Extract results
    Vec3f difference;
    difference.x = vgetq_lane_f32(result, 0);
    difference.y = vgetq_lane_f32(result, 1);
    difference.z = vgetq_lane_f32(result, 2);
    
    return difference;
}

/**
 * Performs fast scalar multiplication of a 3D vector using NEON intrinsics
 * @param vec Vector to scale
 * @param scalar Scalar value to multiply by
 * @return Scaled vector
 */
inline Vec3f Multiply3(const Vec3f& vec, float scalar) {
    // Load vector components into NEON register
    float32x4_t v = vdupq_n_f32(0.0f);
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);
    
    // Create scalar register with all lanes set to scalar value
    float32x4_t s = vdupq_n_f32(scalar);
    
    // Multiply vector by scalar
    float32x4_t result = vmulq_f32(v, s);
    
    // Extract results
    Vec3f scaled;
    scaled.x = vgetq_lane_f32(result, 0);
    scaled.y = vgetq_lane_f32(result, 1);
    scaled.z = vgetq_lane_f32(result, 2);
    
    return scaled;
}

/**
 * Performs a fast linear interpolation between two 3D vectors using NEON intrinsics
 * @param start Start vector
 * @param end End vector
 * @param t Interpolation factor (0.0 to 1.0)
 * @return Interpolated vector
 */
inline Vec3f Lerp3(const Vec3f& start, const Vec3f& end, float t) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    v1 = vsetq_lane_f32(start.x, v1, 0);
    v1 = vsetq_lane_f32(start.y, v1, 1);
    v1 = vsetq_lane_f32(start.z, v1, 2);
    
    v2 = vsetq_lane_f32(end.x, v2, 0);
    v2 = vsetq_lane_f32(end.y, v2, 1);
    v2 = vsetq_lane_f32(end.z, v2, 2);
    
    // Create t register with all lanes set to t value
    float32x4_t t_vec = vdupq_n_f32(t);
    
    // Create (1-t) register
    float32x4_t one_minus_t = vsubq_f32(vdupq_n_f32(1.0f), t_vec);
    
    // Compute v1 * (1-t) + v2 * t
    float32x4_t v1_scaled = vmulq_f32(v1, one_minus_t);
    float32x4_t v2_scaled = vmulq_f32(v2, t_vec);
    float32x4_t result = vaddq_f32(v1_scaled, v2_scaled);
    
    // Extract results
    Vec3f lerp;
    lerp.x = vgetq_lane_f32(result, 0);
    lerp.y = vgetq_lane_f32(result, 1);
    lerp.z = vgetq_lane_f32(result, 2);
    
    return lerp;
}

} // namespace NEON
} // namespace Common

#endif // defined(__ARM_NEON) || defined(__aarch64__)
