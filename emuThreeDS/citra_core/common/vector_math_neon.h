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

/**
 * Performs fast vector division of a 3D vector by a scalar using NEON intrinsics
 * @param vec Vector to divide
 * @param scalar Scalar value to divide by
 * @return Divided vector
 */
inline Vec3f Divide3(const Vec3f& vec, float scalar) {
    // Load vector components into NEON registers
    float32x4_t v = vdupq_n_f32(0.0f);
    
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);
    
    // Create reciprocal of scalar
    float32x4_t recip = vdupq_n_f32(1.0f / scalar);
    
    // Multiply by reciprocal (faster than division)
    float32x4_t result = vmulq_f32(v, recip);
    
    // Extract results
    Vec3f divided;
    divided.x = vgetq_lane_f32(result, 0);
    divided.y = vgetq_lane_f32(result, 1);
    divided.z = vgetq_lane_f32(result, 2);
    
    return divided;
}

/**
 * Performs a fast equality check between two 3D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return True if vectors are equal, false otherwise
 */
inline bool Equals3(const Vec3f& vec1, const Vec3f& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    
    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    
    // Compare for equality
    uint32x4_t cmp = vceqq_f32(v1, v2);
    
    // Check if all components are equal
    uint32x2_t and_narrow = vand_u32(vget_low_u32(cmp), vget_high_u32(cmp));
    uint32x2_t folded = vand_u32(and_narrow, vrev64_u32(and_narrow));
    
    // Extract result (true if all components equal)
    return vget_lane_u32(folded, 0) != 0;
}

/**
 * Performs a fast inequality check between two 3D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return True if vectors are not equal, false otherwise
 */
inline bool NotEquals3(const Vec3f& vec1, const Vec3f& vec2) {
    return !Equals3(vec1, vec2);
}

/**
 * Computes the squared length (magnitude) of a 3D vector using NEON intrinsics
 * @param vec Vector to compute squared length for
 * @return Vector squared length
 */
inline float Length2_3(const Vec3f& vec) {
    // Load vector components into NEON registers
    float32x4_t v = vdupq_n_f32(0.0f);
    
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);
    
    // Square each component
    float32x4_t squared = vmulq_f32(v, v);
    
    // Sum components horizontally
    float32x2_t sum = vpadd_f32(vget_low_f32(squared), vget_high_f32(squared));
    sum = vpadd_f32(sum, sum);
    
    // Extract result
    return vget_lane_f32(sum, 0);
}

/**
 * Performs fast vector addition of two 4D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return Sum vector
 */
inline Vec4f Add4(const Vec4f& vec1, const Vec4f& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    v1 = vsetq_lane_f32(vec1.w, v1, 3);
    
    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    v2 = vsetq_lane_f32(vec2.w, v2, 3);
    
    // Add vectors component-wise
    float32x4_t result = vaddq_f32(v1, v2);
    
    // Extract results
    Vec4f sum;
    sum.x = vgetq_lane_f32(result, 0);
    sum.y = vgetq_lane_f32(result, 1);
    sum.z = vgetq_lane_f32(result, 2);
    sum.w = vgetq_lane_f32(result, 3);
    
    return sum;
}

/**
 * Performs fast vector subtraction of two 4D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return Difference vector
 */
inline Vec4f Subtract4(const Vec4f& vec1, const Vec4f& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    v1 = vsetq_lane_f32(vec1.w, v1, 3);
    
    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    v2 = vsetq_lane_f32(vec2.w, v2, 3);
    
    // Subtract vectors component-wise
    float32x4_t result = vsubq_f32(v1, v2);
    
    // Extract results
    Vec4f diff;
    diff.x = vgetq_lane_f32(result, 0);
    diff.y = vgetq_lane_f32(result, 1);
    diff.z = vgetq_lane_f32(result, 2);
    diff.w = vgetq_lane_f32(result, 3);
    
    return diff;
}

/**
 * Performs fast scalar multiplication of a 4D vector using NEON intrinsics
 * @param vec Vector to scale
 * @param scalar Scalar value to multiply by
 * @return Scaled vector
 */
inline Vec4<float> Multiply4(const Vec4<float>& vec, float scalar) {
    // Load vector components into NEON registers
    float32x4_t v = vdupq_n_f32(0.0f);
    
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);
    v = vsetq_lane_f32(vec.w, v, 3);
    
    // Create scalar register with all lanes set to scalar value
    float32x4_t s = vdupq_n_f32(scalar);
    
    // Multiply vectors component-wise
    float32x4_t result = vmulq_f32(v, s);
    
    // Extract results
    Vec4<float> scaled;
    scaled.x = vgetq_lane_f32(result, 0);
    scaled.y = vgetq_lane_f32(result, 1);
    scaled.z = vgetq_lane_f32(result, 2);
    scaled.w = vgetq_lane_f32(result, 3);
    
    return scaled;
}

/**
 * Performs fast vector division of a 4D vector by a scalar using NEON intrinsics
 * @param vec Vector to divide
 * @param scalar Scalar value to divide by
 * @return Divided vector
 */
inline Vec4f Divide4(const Vec4f& vec, float scalar) {
    // Load vector components into NEON registers
    float32x4_t v = vdupq_n_f32(0.0f);
    
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);
    v = vsetq_lane_f32(vec.w, v, 3);
    
    // Create reciprocal of scalar
    float32x4_t recip = vdupq_n_f32(1.0f / scalar);
    
    // Multiply by reciprocal (faster than division)
    float32x4_t result = vmulq_f32(v, recip);
    
    // Extract results
    Vec4f divided;
    divided.x = vgetq_lane_f32(result, 0);
    divided.y = vgetq_lane_f32(result, 1);
    divided.z = vgetq_lane_f32(result, 2);
    divided.w = vgetq_lane_f32(result, 3);
    
    return divided;
}

/**
 * Performs a fast equality check between two 4D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return True if vectors are equal, false otherwise
 */
inline bool Equals4(const Vec4f& vec1, const Vec4f& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    v1 = vsetq_lane_f32(vec1.w, v1, 3);
    
    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    v2 = vsetq_lane_f32(vec2.w, v2, 3);
    
    // Compare for equality
    uint32x4_t cmp = vceqq_f32(v1, v2);
    
    // Check if all components are equal
    uint8x8_t and_narrow = vqmovn_u16(vcombine_u16(vmovn_u32(cmp), vmovn_u32(cmp)));
    uint64x1_t and_val = vreinterpret_u64_u8(and_narrow);
    
    return (vget_lane_u64(and_val, 0) == 0xFFFFFFFFFFFFFFFF);
}

/**
 * Performs a fast inequality check between two 4D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return True if vectors are not equal, false otherwise
 */
inline bool NotEquals4(const Vec4f& vec1, const Vec4f& vec2) {
    return !Equals4(vec1, vec2);
}

/**
 * Computes the squared length (magnitude) of a 4D vector using NEON intrinsics
 * @param vec Vector to compute squared length for
 * @return Vector squared length
 */
inline float Length2_4(const Vec4f& vec) {
    // Load vector components into NEON registers
    float32x4_t v = vdupq_n_f32(0.0f);
    
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);
    v = vsetq_lane_f32(vec.w, v, 3);
    
    // Multiply vector by itself component-wise
    float32x4_t squared = vmulq_f32(v, v);
    
    // Sum components horizontally
    float32x2_t sum = vpadd_f32(vget_low_f32(squared), vget_high_f32(squared));
    sum = vpadd_f32(sum, sum);
    
    // Extract result
    return vget_lane_f32(sum, 0);
}

/**
 * Computes the length (magnitude) of a 4D vector using NEON intrinsics
 * @param vec Vector to compute length for
 * @return Vector length
 */
inline float Length4(const Vec4f& vec) {
    float length_squared = Length2_4(vec);
    return std::sqrt(length_squared);
}

/**
 * Performs bilinear interpolation on 4D vectors using NEON intrinsics
 * @param x00 First vector (0,0)
 * @param x01 Second vector (0,1)
 * @param x10 Third vector (1,0)
 * @param x11 Fourth vector (1,1)
 * @param s Interpolation parameter for first dimension
 * @param t Interpolation parameter for second dimension
 * @return Interpolated vector
 */
inline Vec4f BilinearInterp4Fast(const Vec4f& x00, const Vec4f& x01, const Vec4f& x10, const Vec4f& x11,
                              const float s, const float t) {
    // Load vector components into NEON registers
    float32x4_t v00 = vdupq_n_f32(0.0f);
    float32x4_t v01 = vdupq_n_f32(0.0f);
    float32x4_t v10 = vdupq_n_f32(0.0f);
    float32x4_t v11 = vdupq_n_f32(0.0f);
    
    // Load x00
    v00 = vsetq_lane_f32(x00.x, v00, 0);
    v00 = vsetq_lane_f32(x00.y, v00, 1);
    v00 = vsetq_lane_f32(x00.z, v00, 2);
    v00 = vsetq_lane_f32(x00.w, v00, 3);
    
    // Load x01
    v01 = vsetq_lane_f32(x01.x, v01, 0);
    v01 = vsetq_lane_f32(x01.y, v01, 1);
    v01 = vsetq_lane_f32(x01.z, v01, 2);
    v01 = vsetq_lane_f32(x01.w, v01, 3);
    
    // Load x10
    v10 = vsetq_lane_f32(x10.x, v10, 0);
    v10 = vsetq_lane_f32(x10.y, v10, 1);
    v10 = vsetq_lane_f32(x10.z, v10, 2);
    v10 = vsetq_lane_f32(x10.w, v10, 3);
    
    // Load x11
    v11 = vsetq_lane_f32(x11.x, v11, 0);
    v11 = vsetq_lane_f32(x11.y, v11, 1);
    v11 = vsetq_lane_f32(x11.z, v11, 2);
    v11 = vsetq_lane_f32(x11.w, v11, 3);
    
    // Create s and (1-s) scalars
    float32x4_t s_vec = vdupq_n_f32(s);
    float32x4_t one_minus_s = vdupq_n_f32(1.0f - s);
    
    // Create t and (1-t) scalars
    float32x4_t t_vec = vdupq_n_f32(t);
    float32x4_t one_minus_t = vdupq_n_f32(1.0f - t);
    
    // First interpolation: y0 = x00 * (1-s) + x01 * s
    float32x4_t y0 = vmlaq_f32(vmulq_f32(v00, one_minus_s), v01, s_vec);
    
    // Second interpolation: y1 = x10 * (1-s) + x11 * s
    float32x4_t y1 = vmlaq_f32(vmulq_f32(v10, one_minus_s), v11, s_vec);
    
    // Final interpolation: result = y0 * (1-t) + y1 * t
    float32x4_t result = vmlaq_f32(vmulq_f32(y0, one_minus_t), y1, t_vec);
    
    // Extract results
    Vec4f interp;
    interp.x = vgetq_lane_f32(result, 0);
    interp.y = vgetq_lane_f32(result, 1);
    interp.z = vgetq_lane_f32(result, 2);
    interp.w = vgetq_lane_f32(result, 3);
    
    return interp;
}

// Normalize3 is already defined above

/**
 * Normalizes a 4D vector using NEON intrinsics
 * @param vec Vector to normalize
 * @return Normalized vector
 */
inline Vec4f Normalize4(const Vec4f& vec) {
    // Calculate length
    float length = Length4(vec);
    
    // Return normalized vector
    if (length > 0.0f) {
        return vec / length;
    }
    
    return vec;
}

/**
 * Normalizes a 3D vector in-place using NEON intrinsics
 * @param vec Vector to normalize
 * @return Length of the vector before normalization
 */
inline float Normalize3Inplace(Vec3f& vec) {
    // Calculate length
    float length = Length3(vec);
    
    // Normalize the vector in-place
    if (length > 0.0f) {
        float32x4_t v_vec = vdupq_n_f32(0.0f);
        v_vec = vsetq_lane_f32(vec.x, v_vec, 0);
        v_vec = vsetq_lane_f32(vec.y, v_vec, 1);
        v_vec = vsetq_lane_f32(vec.z, v_vec, 2);
        
        float32x4_t v_length = vdupq_n_f32(length);
        float32x4_t v_normalized = vdivq_f32(v_vec, v_length);
        
        vec.x = vgetq_lane_f32(v_normalized, 0);
        vec.y = vgetq_lane_f32(v_normalized, 1);
        vec.z = vgetq_lane_f32(v_normalized, 2);
    }
    
    return length;
}

/**
 * Normalizes a 4D vector in-place using NEON intrinsics
 * @param vec Vector to normalize
 * @return Length of the vector before normalization
 */
inline float Normalize4Inplace(Vec4f& vec) {
    // Calculate length
    float length = Length4(vec);
    
    // Normalize the vector in-place
    if (length > 0.0f) {
        float32x4_t v_vec = vdupq_n_f32(0.0f);
        v_vec = vsetq_lane_f32(vec.x, v_vec, 0);
        v_vec = vsetq_lane_f32(vec.y, v_vec, 1);
        v_vec = vsetq_lane_f32(vec.z, v_vec, 2);
        v_vec = vsetq_lane_f32(vec.w, v_vec, 3);
        
        float32x4_t v_length = vdupq_n_f32(length);
        float32x4_t v_normalized = vdivq_f32(v_vec, v_length);
        
        vec.x = vgetq_lane_f32(v_normalized, 0);
        vec.y = vgetq_lane_f32(v_normalized, 1);
        vec.z = vgetq_lane_f32(v_normalized, 2);
        vec.w = vgetq_lane_f32(v_normalized, 3);
    }
    
    return length;
}

/**
 * Performs linear interpolation between two 4D vectors using NEON intrinsics
 * @param begin Starting vector
 * @param end Ending vector
 * @param t Interpolation parameter (0.0 = begin, 1.0 = end)
 * @return Interpolated vector
 */
inline Vec4f Lerp4(const Vec4f& begin, const Vec4f& end, float t) {
    // Load vector components into NEON registers
    float32x4_t v_begin = vdupq_n_f32(0.0f);
    float32x4_t v_end = vdupq_n_f32(0.0f);
    
    // Load begin vector
    v_begin = vsetq_lane_f32(begin.x, v_begin, 0);
    v_begin = vsetq_lane_f32(begin.y, v_begin, 1);
    v_begin = vsetq_lane_f32(begin.z, v_begin, 2);
    v_begin = vsetq_lane_f32(begin.w, v_begin, 3);
    
    // Load end vector
    v_end = vsetq_lane_f32(end.x, v_end, 0);
    v_end = vsetq_lane_f32(end.y, v_end, 1);
    v_end = vsetq_lane_f32(end.z, v_end, 2);
    v_end = vsetq_lane_f32(end.w, v_end, 3);
    
    // Create t and (1-t) scalars
    float32x4_t t_vec = vdupq_n_f32(t);
    float32x4_t one_minus_t = vdupq_n_f32(1.0f - t);
    
    // Compute the interpolation: begin * (1-t) + end * t
    float32x4_t result = vmlaq_f32(vmulq_f32(v_begin, one_minus_t), v_end, t_vec);
    
    // Extract results
    Vec4f interp;
    interp.x = vgetq_lane_f32(result, 0);
    interp.y = vgetq_lane_f32(result, 1);
    interp.z = vgetq_lane_f32(result, 2);
    interp.w = vgetq_lane_f32(result, 3);
    
    return interp;
}

/**
 * Performs a fast dot product between two 2D vectors using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return Dot product result
 */
inline float Dot2(const Vec2f& vec1, const Vec2f& vec2) {
    // Load vector components into NEON registers
    float32x2_t v1 = vdup_n_f32(0.0f);
    float32x2_t v2 = vdup_n_f32(0.0f);
    
    v1 = vset_lane_f32(vec1.x, v1, 0);
    v1 = vset_lane_f32(vec1.y, v1, 1);
    
    v2 = vset_lane_f32(vec2.x, v2, 0);
    v2 = vset_lane_f32(vec2.y, v2, 1);
    
    // Multiply vectors component-wise
    float32x2_t mul = vmul_f32(v1, v2);
    
    // Sum components horizontally
    float32x2_t sum = vpadd_f32(mul, mul);
    
    // Extract result
    return vget_lane_f32(sum, 0);
}

/**
 * Performs bilinear interpolation using NEON intrinsics
 * @param x00 Value at (0,0)
 * @param x01 Value at (0,1)
 * @param x10 Value at (1,0)
 * @param x11 Value at (1,1)
 * @param s Interpolation factor along first axis (0.0 to 1.0)
 * @param t Interpolation factor along second axis (0.0 to 1.0)
 * @return Interpolated value
 */
inline Vec3f BilinearInterp3(const Vec3f& x00, const Vec3f& x01, const Vec3f& x10, const Vec3f& x11,
                           float s, float t) {
    // First interpolate along s (x direction)
    Vec3f y0 = Lerp3(x00, x01, s);
    Vec3f y1 = Lerp3(x10, x11, s);
    
    // Then interpolate the results along t (y direction)
    return Lerp3(y0, y1, t);
}

/**
 * Performs bilinear interpolation using NEON intrinsics (optimized version)
 * @param x00 Value at (0,0)
 * @param x01 Value at (0,1)
 * @param x10 Value at (1,0)
 * @param x11 Value at (1,1)
 * @param s Interpolation factor along first axis (0.0 to 1.0)
 * @param t Interpolation factor along second axis (0.0 to 1.0)
 * @return Interpolated value
 */
inline Vec3f BilinearInterp3Fast(const Vec3f& x00, const Vec3f& x01, const Vec3f& x10, const Vec3f& x11,
                               float s, float t) {
    // Load all four vectors into NEON registers
    float32x4_t v00 = vdupq_n_f32(0.0f);
    float32x4_t v01 = vdupq_n_f32(0.0f);
    float32x4_t v10 = vdupq_n_f32(0.0f);
    float32x4_t v11 = vdupq_n_f32(0.0f);
    
    v00 = vsetq_lane_f32(x00.x, v00, 0);
    v00 = vsetq_lane_f32(x00.y, v00, 1);
    v00 = vsetq_lane_f32(x00.z, v00, 2);
    
    v01 = vsetq_lane_f32(x01.x, v01, 0);
    v01 = vsetq_lane_f32(x01.y, v01, 1);
    v01 = vsetq_lane_f32(x01.z, v01, 2);
    
    v10 = vsetq_lane_f32(x10.x, v10, 0);
    v10 = vsetq_lane_f32(x10.y, v10, 1);
    v10 = vsetq_lane_f32(x10.z, v10, 2);
    
    v11 = vsetq_lane_f32(x11.x, v11, 0);
    v11 = vsetq_lane_f32(x11.y, v11, 1);
    v11 = vsetq_lane_f32(x11.z, v11, 2);
    
    // Create s and t registers
    float32x4_t s_vec = vdupq_n_f32(s);
    float32x4_t t_vec = vdupq_n_f32(t);
    float32x4_t one_minus_s = vsubq_f32(vdupq_n_f32(1.0f), s_vec);
    float32x4_t one_minus_t = vsubq_f32(vdupq_n_f32(1.0f), t_vec);
    
    // Compute bilinear interpolation in one go:
    // result = x00 * (1-s) * (1-t) + x01 * (1-s) * t + x10 * s * (1-t) + x11 * s * t
    float32x4_t w00 = vmulq_f32(vmulq_f32(v00, one_minus_s), one_minus_t);
    float32x4_t w01 = vmulq_f32(vmulq_f32(v01, one_minus_s), t_vec);
    float32x4_t w10 = vmulq_f32(vmulq_f32(v10, s_vec), one_minus_t);
    float32x4_t w11 = vmulq_f32(vmulq_f32(v11, s_vec), t_vec);
    
    // Sum all weighted values
    float32x4_t result = vaddq_f32(vaddq_f32(w00, w01), vaddq_f32(w10, w11));
    
    // Extract results
    Vec3f interp;
    interp.x = vgetq_lane_f32(result, 0);
    interp.y = vgetq_lane_f32(result, 1);
    interp.z = vgetq_lane_f32(result, 2);
    
    return interp;
}

/**
 * Multiplies two 3D vectors component-wise using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return Component-wise product
 */
inline Vec3<float> Multiply3(const Vec3<float>& vec1, const Vec3<float>& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    
    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    
    // Multiply vectors component-wise
    float32x4_t result = vmulq_f32(v1, v2);
    
    // Extract results
    Vec3<float> product;
    product.x = vgetq_lane_f32(result, 0);
    product.y = vgetq_lane_f32(result, 1);
    product.z = vgetq_lane_f32(result, 2);
    
    return product;
}

/**
 * Multiplies two 4D vectors component-wise using NEON intrinsics
 * @param vec1 First vector
 * @param vec2 Second vector
 * @return Component-wise product
 */
inline Vec4<float> Multiply4(const Vec4<float>& vec1, const Vec4<float>& vec2) {
    // Load vector components into NEON registers
    float32x4_t v1 = vdupq_n_f32(0.0f);
    float32x4_t v2 = vdupq_n_f32(0.0f);
    
    v1 = vsetq_lane_f32(vec1.x, v1, 0);
    v1 = vsetq_lane_f32(vec1.y, v1, 1);
    v1 = vsetq_lane_f32(vec1.z, v1, 2);
    v1 = vsetq_lane_f32(vec1.w, v1, 3);
    
    v2 = vsetq_lane_f32(vec2.x, v2, 0);
    v2 = vsetq_lane_f32(vec2.y, v2, 1);
    v2 = vsetq_lane_f32(vec2.z, v2, 2);
    v2 = vsetq_lane_f32(vec2.w, v2, 3);
    
    // Multiply vectors component-wise
    float32x4_t result = vmulq_f32(v1, v2);
    
    // Extract results
    Vec4<float> product;
    product.x = vgetq_lane_f32(result, 0);
    product.y = vgetq_lane_f32(result, 1);
    product.z = vgetq_lane_f32(result, 2);
    product.w = vgetq_lane_f32(result, 3);
    
    return product;
}

/**
 * Multiplies a 3D vector by a scalar using NEON intrinsics
 * @param vec Vector to multiply
 * @param scalar Scalar value to multiply by
 * @return Scaled vector
 */
inline Vec3<float> Multiply3Scalar(const Vec3<float>& vec, float scalar) {
    // Load vector components into NEON registers
    float32x4_t v = vdupq_n_f32(0.0f);
    
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);
    
    // Create scalar vector
    float32x4_t s = vdupq_n_f32(scalar);
    
    // Multiply vector by scalar
    float32x4_t result = vmulq_f32(v, s);
    
    // Extract results
    Vec3<float> scaled;
    scaled.x = vgetq_lane_f32(result, 0);
    scaled.y = vgetq_lane_f32(result, 1);
    scaled.z = vgetq_lane_f32(result, 2);
    
    return scaled;
}

/**
 * Multiplies a 4D vector by a scalar using NEON intrinsics
 * @param vec Vector to multiply
 * @param scalar Scalar value to multiply by
 * @return Scaled vector
 */
inline Vec4<float> Multiply4Scalar(const Vec4<float>& vec, float scalar) {
    // Load vector components into NEON registers
    float32x4_t v = vdupq_n_f32(0.0f);
    
    v = vsetq_lane_f32(vec.x, v, 0);
    v = vsetq_lane_f32(vec.y, v, 1);
    v = vsetq_lane_f32(vec.z, v, 2);
    v = vsetq_lane_f32(vec.w, v, 3);
    
    // Create scalar vector
    float32x4_t s = vdupq_n_f32(scalar);
    
    // Multiply vector by scalar
    float32x4_t result = vmulq_f32(v, s);
    
    // Extract results
    Vec4<float> scaled;
    scaled.x = vgetq_lane_f32(result, 0);
    scaled.y = vgetq_lane_f32(result, 1);
    scaled.z = vgetq_lane_f32(result, 2);
    scaled.w = vgetq_lane_f32(result, 3);
    
    return scaled;
}

} // namespace NEON
} // namespace Common

#endif // defined(__ARM_NEON) || defined(__aarch64__)
