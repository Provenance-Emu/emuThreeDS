// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/vector_math.h"
#include "common/math_util.h"
#include "video_core/geometry_culling.h"

// Only use ARM NEON intrinsics on ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

namespace Common {

/**
 * ARM NEON optimized matrix math operations
 * These functions provide SIMD-accelerated implementations of common matrix operations
 * used throughout the emulator, particularly in graphics transformations.
 */
namespace NEON {

/**
 * Performs matrix-vector multiplication using NEON intrinsics
 * @param matrix 4x4 matrix
 * @param vector 4D vector
 * @return Resulting 4D vector
 */
inline Vec4<float> MultiplyMatrixVector(const Pica::Matrix4x4& matrix, const Vec4<float>& vector) {
    // Load vector components into NEON register
    float32x4_t vec = vld1q_f32(&vector.x);

    // Load matrix rows into NEON registers
    float32x4_t row0 = vld1q_f32(&matrix.r[0].x);
    float32x4_t row1 = vld1q_f32(&matrix.r[1].x);
    float32x4_t row2 = vld1q_f32(&matrix.r[2].x);
    float32x4_t row3 = vld1q_f32(&matrix.r[3].x);

    // Multiply each row by the vector and sum components
    float32x4_t result;

    // Multiply row0 by vector
    float32x4_t mul0 = vmulq_f32(row0, vec);
    float32x2_t sum0 = vpadd_f32(vget_low_f32(mul0), vget_high_f32(mul0));
    sum0 = vpadd_f32(sum0, sum0);
    result = vsetq_lane_f32(vget_lane_f32(sum0, 0), result, 0);

    // Multiply row1 by vector
    float32x4_t mul1 = vmulq_f32(row1, vec);
    float32x2_t sum1 = vpadd_f32(vget_low_f32(mul1), vget_high_f32(mul1));
    sum1 = vpadd_f32(sum1, sum1);
    result = vsetq_lane_f32(vget_lane_f32(sum1, 0), result, 1);

    // Multiply row2 by vector
    float32x4_t mul2 = vmulq_f32(row2, vec);
    float32x2_t sum2 = vpadd_f32(vget_low_f32(mul2), vget_high_f32(mul2));
    sum2 = vpadd_f32(sum2, sum2);
    result = vsetq_lane_f32(vget_lane_f32(sum2, 0), result, 2);

    // Multiply row3 by vector
    float32x4_t mul3 = vmulq_f32(row3, vec);
    float32x2_t sum3 = vpadd_f32(vget_low_f32(mul3), vget_high_f32(mul3));
    sum3 = vpadd_f32(sum3, sum3);
    result = vsetq_lane_f32(vget_lane_f32(sum3, 0), result, 3);

    // Create and return the result vector
    Vec4<float> output;
    output.x = vgetq_lane_f32(result, 0);
    output.y = vgetq_lane_f32(result, 1);
    output.z = vgetq_lane_f32(result, 2);
    output.w = vgetq_lane_f32(result, 3);

    return output;
}

/**
 * Performs matrix-vector multiplication for 3x3 matrices using NEON intrinsics
 * @param matrix 3x3 matrix
 * @param vector 3D vector
 * @return Transformed 3D vector
 */
inline Vec3<float> MultiplyMatrixVector(const Pica::Matrix3x3& matrix, const Vec3<float>& vector) {
    // Load vector components into NEON register
    float32x4_t vec = vdupq_n_f32(0.0f);
    vec = vsetq_lane_f32(vector.x, vec, 0);
    vec = vsetq_lane_f32(vector.y, vec, 1);
    vec = vsetq_lane_f32(vector.z, vec, 2);
    
    // Load matrix rows into NEON registers (pad with zeros for 4th column)
    float32x4_t row0 = vdupq_n_f32(0.0f);
    float32x4_t row1 = vdupq_n_f32(0.0f);
    float32x4_t row2 = vdupq_n_f32(0.0f);
    
    // Load first row
    row0 = vsetq_lane_f32(matrix.r[0].x, row0, 0);
    row0 = vsetq_lane_f32(matrix.r[0].y, row0, 1);
    row0 = vsetq_lane_f32(matrix.r[0].z, row0, 2);
    
    // Load second row
    row1 = vsetq_lane_f32(matrix.r[1].x, row1, 0);
    row1 = vsetq_lane_f32(matrix.r[1].y, row1, 1);
    row1 = vsetq_lane_f32(matrix.r[1].z, row1, 2);
    
    // Load third row
    row2 = vsetq_lane_f32(matrix.r[2].x, row2, 0);
    row2 = vsetq_lane_f32(matrix.r[2].y, row2, 1);
    row2 = vsetq_lane_f32(matrix.r[2].z, row2, 2);
    
    // Multiply each row by the vector and sum components
    float32x4_t result = vdupq_n_f32(0.0f);
    
    // Multiply row0 by vector
    float32x4_t mul0 = vmulq_f32(row0, vec);
    float32x2_t sum0 = vpadd_f32(vget_low_f32(mul0), vget_high_f32(mul0));
    sum0 = vpadd_f32(sum0, vdup_n_f32(0.0f));
    result = vsetq_lane_f32(vget_lane_f32(sum0, 0), result, 0);
    
    // Multiply row1 by vector
    float32x4_t mul1 = vmulq_f32(row1, vec);
    float32x2_t sum1 = vpadd_f32(vget_low_f32(mul1), vget_high_f32(mul1));
    sum1 = vpadd_f32(sum1, vdup_n_f32(0.0f));
    result = vsetq_lane_f32(vget_lane_f32(sum1, 0), result, 1);
    
    // Multiply row2 by vector
    float32x4_t mul2 = vmulq_f32(row2, vec);
    float32x2_t sum2 = vpadd_f32(vget_low_f32(mul2), vget_high_f32(mul2));
    sum2 = vpadd_f32(sum2, vdup_n_f32(0.0f));
    result = vsetq_lane_f32(vget_lane_f32(sum2, 0), result, 2);
    
    // Create and return the result vector
    Vec3<float> output;
    output.x = vgetq_lane_f32(result, 0);
    output.y = vgetq_lane_f32(result, 1);
    output.z = vgetq_lane_f32(result, 2);
    
    return output;
}

/**
 * Normalizes a 3D vector using NEON intrinsics
 * @param vector Vector to normalize
 * @return Normalized vector
 */
inline Vec3<float> Normalize(const Vec3<float>& vector) {
    // Load vector into NEON register
    float32x4_t vec = vdupq_n_f32(0.0f);
    vec = vsetq_lane_f32(vector.x, vec, 0);
    vec = vsetq_lane_f32(vector.y, vec, 1);
    vec = vsetq_lane_f32(vector.z, vec, 2);
    
    // Compute squared length
    float32x4_t squared = vmulq_f32(vec, vec);
    float32x2_t sum = vpadd_f32(vget_low_f32(squared), vget_high_f32(squared));
    sum = vpadd_f32(sum, vdup_n_f32(0.0f));
    float squared_length = vget_lane_f32(sum, 0);
    
    // Create result vector
    Vec3<float> result = vector;
    
    // Check if length is non-zero
    if (squared_length > 0.00001f) {
        // Compute reciprocal square root
        float32x2_t rsqrt = vrsqrte_f32(vdup_n_f32(squared_length));
        
        // Refine with Newton-Raphson iterations for better accuracy
        // One iteration: x = x * (1.5 - 0.5 * a * x * x)
        float32x2_t rsqrt_step = vmul_f32(rsqrt, vrsqrts_f32(vdup_n_f32(squared_length), vmul_f32(rsqrt, rsqrt)));
        
        // Second iteration for better accuracy
        rsqrt_step = vmul_f32(rsqrt_step, vrsqrts_f32(vdup_n_f32(squared_length), vmul_f32(rsqrt_step, rsqrt_step)));
        
        // Scale vector by reciprocal square root
        float scale = vget_lane_f32(rsqrt_step, 0);
        float32x4_t normalized = vmulq_n_f32(vec, scale);
        
        // Store result
        result.x = vgetq_lane_f32(normalized, 0);
        result.y = vgetq_lane_f32(normalized, 1);
        result.z = vgetq_lane_f32(normalized, 2);
    }
    
    return result;
}

/**
 * Transforms a 3D vector by a 4x4 matrix using NEON intrinsics
 * This is optimized for the common case of transforming a point or direction
 * @param matrix 4x4 transformation matrix
 * @param vector 3D vector to transform
 * @param is_position True if vector is a position (w=1), false if direction (w=0)
 * @return Transformed 3D vector
 */
inline Vec3<float> TransformVec3(const Pica::Matrix4x4& matrix, const Vec3<float>& vector, bool is_position) {
    // Create 4D vector from 3D vector
    float32x4_t vec = vdupq_n_f32(0.0f);
    vec = vsetq_lane_f32(vector.x, vec, 0);
    vec = vsetq_lane_f32(vector.y, vec, 1);
    vec = vsetq_lane_f32(vector.z, vec, 2);
    vec = vsetq_lane_f32(is_position ? 1.0f : 0.0f, vec, 3);
    
    // Load matrix rows into NEON registers
    float32x4_t row0 = vld1q_f32(&matrix.r[0].x);
    float32x4_t row1 = vld1q_f32(&matrix.r[1].x);
    float32x4_t row2 = vld1q_f32(&matrix.r[2].x);
    float32x4_t row3 = vld1q_f32(&matrix.r[3].x);
    
    // Multiply each row by the vector and sum components
    float32x4_t result;
    
    // Multiply row0 by vector
    float32x4_t mul0 = vmulq_f32(row0, vec);
    float32x2_t sum0 = vpadd_f32(vget_low_f32(mul0), vget_high_f32(mul0));
    sum0 = vpadd_f32(sum0, sum0);
    result = vsetq_lane_f32(vget_lane_f32(sum0, 0), result, 0);
    
    // Multiply row1 by vector
    float32x4_t mul1 = vmulq_f32(row1, vec);
    float32x2_t sum1 = vpadd_f32(vget_low_f32(mul1), vget_high_f32(mul1));
    sum1 = vpadd_f32(sum1, sum1);
    result = vsetq_lane_f32(vget_lane_f32(sum1, 0), result, 1);
    
    // Multiply row2 by vector
    float32x4_t mul2 = vmulq_f32(row2, vec);
    float32x2_t sum2 = vpadd_f32(vget_low_f32(mul2), vget_high_f32(mul2));
    sum2 = vpadd_f32(sum2, sum2);
    result = vsetq_lane_f32(vget_lane_f32(sum2, 0), result, 2);
    
    // Multiply row3 by vector (for w component)
    float32x4_t mul3 = vmulq_f32(row3, vec);
    float32x2_t sum3 = vpadd_f32(vget_low_f32(mul3), vget_high_f32(mul3));
    sum3 = vpadd_f32(sum3, sum3);
    result = vsetq_lane_f32(vget_lane_f32(sum3, 0), result, 3);
    
    // Perform perspective division if w != 0
    float w = vgetq_lane_f32(result, 3);
    if (is_position && std::abs(w) > 0.00001f) {
        float32x4_t w_recip = vdupq_n_f32(1.0f / w);
        result = vmulq_f32(result, w_recip);
    }
    
    // Extract results (only x, y, z)
    Vec3<float> output;
    output.x = vgetq_lane_f32(result, 0);
    output.y = vgetq_lane_f32(result, 1);
    output.z = vgetq_lane_f32(result, 2);
    
    return output;
}

/**
 * Computes the inverse of a 4x4 matrix using NEON intrinsics
 * @param matrix Matrix to invert
 * @return Inverted matrix
 */
inline Pica::Matrix4x4 InvertMatrix4x4(const Pica::Matrix4x4& matrix) {
    Pica::Matrix4x4 result;
    
    // Load matrix rows into NEON registers
    float32x4_t row0 = vld1q_f32(&matrix.r[0].x);
    float32x4_t row1 = vld1q_f32(&matrix.r[1].x);
    float32x4_t row2 = vld1q_f32(&matrix.r[2].x);
    float32x4_t row3 = vld1q_f32(&matrix.r[3].x);
    
    // Create a transposed version of the matrix using direct element access
    // instead of dynamic indexing with vgetq_lane_f32/vsetq_lane_f32
    float32x4_t col0 = {
        matrix.r[0].x, matrix.r[1].x, matrix.r[2].x, matrix.r[3].x
    };
    float32x4_t col1 = {
        matrix.r[0].y, matrix.r[1].y, matrix.r[2].y, matrix.r[3].y
    };
    float32x4_t col2 = {
        matrix.r[0].z, matrix.r[1].z, matrix.r[2].z, matrix.r[3].z
    };
    float32x4_t col3 = {
        matrix.r[0].w, matrix.r[1].w, matrix.r[2].w, matrix.r[3].w
    };
    
    // Compute matrix of cofactors using the classical adjoint formula
    
    // Compute cofactors for first row
    // For M_00: det of the 3x3 submatrix formed by removing row 0 and column 0
    float m_11 = matrix.r[1].y;
    float m_12 = matrix.r[1].z;
    float m_13 = matrix.r[1].w;
    float m_21 = matrix.r[2].y;
    float m_22 = matrix.r[2].z;
    float m_23 = matrix.r[2].w;
    float m_31 = matrix.r[3].y;
    float m_32 = matrix.r[3].z;
    float m_33 = matrix.r[3].w;
    
    float c_00 = m_11 * (m_22 * m_33 - m_23 * m_32) - 
                m_12 * (m_21 * m_33 - m_23 * m_31) + 
                m_13 * (m_21 * m_32 - m_22 * m_31);
    
    // For M_01: -det of the 3x3 submatrix formed by removing row 0 and column 1
    float m_10 = matrix.r[1].x;
    float m_20 = matrix.r[2].x;
    float m_30 = matrix.r[3].x;
    
    float c_01 = -(m_10 * (m_22 * m_33 - m_23 * m_32) - 
                 m_12 * (m_20 * m_33 - m_23 * m_30) + 
                 m_13 * (m_20 * m_32 - m_22 * m_30));
    
    // For M_02: det of the 3x3 submatrix formed by removing row 0 and column 2
    float c_02 = m_10 * (m_21 * m_33 - m_23 * m_31) - 
                m_11 * (m_20 * m_33 - m_23 * m_30) + 
                m_13 * (m_20 * m_31 - m_21 * m_30);
    
    // For M_03: -det of the 3x3 submatrix formed by removing row 0 and column 3
    float c_03 = -(m_10 * (m_21 * m_32 - m_22 * m_31) - 
                 m_11 * (m_20 * m_32 - m_22 * m_30) + 
                 m_12 * (m_20 * m_31 - m_21 * m_30));
    
    // Compute cofactors for second row
    float m_00 = matrix.r[0].x;
    float m_01 = matrix.r[0].y;
    float m_02 = matrix.r[0].z;
    float m_03 = matrix.r[0].w;
    
    float c_10 = -(m_01 * (m_22 * m_33 - m_23 * m_32) - 
                 m_02 * (m_21 * m_33 - m_23 * m_31) + 
                 m_03 * (m_21 * m_32 - m_22 * m_31));
    
    float c_11 = m_00 * (m_22 * m_33 - m_23 * m_32) - 
                m_02 * (m_20 * m_33 - m_23 * m_30) + 
                m_03 * (m_20 * m_32 - m_22 * m_30);
    
    float c_12 = -(m_00 * (m_21 * m_33 - m_23 * m_31) - 
                 m_01 * (m_20 * m_33 - m_23 * m_30) + 
                 m_03 * (m_20 * m_31 - m_21 * m_30));
    
    float c_13 = m_00 * (m_21 * m_32 - m_22 * m_31) - 
                m_01 * (m_20 * m_32 - m_22 * m_30) + 
                m_02 * (m_20 * m_31 - m_21 * m_30);
    
    // Compute cofactors for third row
    float c_20 = m_01 * (m_12 * m_33 - m_13 * m_32) - 
                m_02 * (m_11 * m_33 - m_13 * m_31) + 
                m_03 * (m_11 * m_32 - m_12 * m_31);
    
    float c_21 = -(m_00 * (m_12 * m_33 - m_13 * m_32) - 
                 m_02 * (m_10 * m_33 - m_13 * m_30) + 
                 m_03 * (m_10 * m_32 - m_12 * m_30));
    
    float c_22 = m_00 * (m_11 * m_33 - m_13 * m_31) - 
                m_01 * (m_10 * m_33 - m_13 * m_30) + 
                m_03 * (m_10 * m_31 - m_11 * m_30);
    
    float c_23 = -(m_00 * (m_11 * m_32 - m_12 * m_31) - 
                 m_01 * (m_10 * m_32 - m_12 * m_30) + 
                 m_02 * (m_10 * m_31 - m_11 * m_30));
    
    // Compute cofactors for fourth row
    float c_30 = -(m_01 * (m_12 * m_23 - m_13 * m_22) - 
                 m_02 * (m_11 * m_23 - m_13 * m_21) + 
                 m_03 * (m_11 * m_22 - m_12 * m_21));
    
    float c_31 = m_00 * (m_12 * m_23 - m_13 * m_22) - 
                m_02 * (m_10 * m_23 - m_13 * m_20) + 
                m_03 * (m_10 * m_22 - m_12 * m_20);
    
    float c_32 = -(m_00 * (m_11 * m_23 - m_13 * m_21) - 
                 m_01 * (m_10 * m_23 - m_13 * m_20) + 
                 m_03 * (m_10 * m_21 - m_11 * m_20));
    
    float c_33 = m_00 * (m_11 * m_22 - m_12 * m_21) - 
                m_01 * (m_10 * m_22 - m_12 * m_20) + 
                m_02 * (m_10 * m_21 - m_11 * m_20);
    
    // Compute determinant using the first row of cofactors
    float det_value = m_00 * c_00 + m_01 * c_01 + m_02 * c_02 + m_03 * c_03;
    
    // Compute reciprocal of determinant
    float inv_det = 1.0f / det_value;
    
    // Multiply cofactors by reciprocal of determinant to get the inverse
    result.r[0] = {c_00 * inv_det, c_01 * inv_det, c_02 * inv_det, c_03 * inv_det};
    result.r[1] = {c_10 * inv_det, c_11 * inv_det, c_12 * inv_det, c_13 * inv_det};
    result.r[2] = {c_20 * inv_det, c_21 * inv_det, c_22 * inv_det, c_23 * inv_det};
    result.r[3] = {c_30 * inv_det, c_31 * inv_det, c_32 * inv_det, c_33 * inv_det};
    
    return result;
}

} // namespace NEON
} // namespace Common

#endif // defined(__ARM_NEON) || defined(__aarch64__)
