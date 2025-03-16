// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cmath>
#include "video_core/geometry_culling.h"
#include "common/vector_math_neon.h"
#include "common/matrix_math_neon.h"

// Use ARM NEON intrinsics for ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace Pica {

bool GeometryCulling::IsBoundingBoxVisible(const CullingBoundingBox& bbox,
                                           const CullingFrustumPlanes& frustum) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Optimized implementation for ARM64 devices
    // Preload bounding box min/max into NEON registers for faster access
    float32x4_t bbox_min = vdupq_n_f32(0.0f);
    float32x4_t bbox_max = vdupq_n_f32(0.0f);
    
    // Load XYZ components more efficiently
    bbox_min = vsetq_lane_f32(bbox.min.x, bbox_min, 0);
    bbox_min = vsetq_lane_f32(bbox.min.y, bbox_min, 1);
    bbox_min = vsetq_lane_f32(bbox.min.z, bbox_min, 2);
    bbox_min = vsetq_lane_f32(1.0f, bbox_min, 3);  // w component = 1.0f
    
    bbox_max = vsetq_lane_f32(bbox.max.x, bbox_max, 0);
    bbox_max = vsetq_lane_f32(bbox.max.y, bbox_max, 1);
    bbox_max = vsetq_lane_f32(bbox.max.z, bbox_max, 2);
    bbox_max = vsetq_lane_f32(1.0f, bbox_max, 3);  // w component = 1.0f
    
    // Process frustum planes in batches of 2 for better NEON utilization
    // This reduces branch mispredictions and improves instruction pipelining
    for (int i = 0; i < 6; i += 2) {
        // Load two plane coefficients at once when possible
        float32x4_t plane_vec1 = vld1q_f32(&frustum.planes[i].x);
        float32x4_t plane_vec2 = (i + 1 < 6) ? vld1q_f32(&frustum.planes[i + 1].x) : vdupq_n_f32(0.0f);
        
        // Create masks to select between min and max based on plane normals
        uint32x4_t mask1 = vcgtq_f32(plane_vec1, vdupq_n_f32(0.0f));
        uint32x4_t mask2 = vcgtq_f32(plane_vec2, vdupq_n_f32(0.0f));
        
        // Select p-vertex components (furthest points in normal directions)
        float32x4_t p_vertex1 = vbslq_f32(mask1, bbox_max, bbox_min);
        float32x4_t p_vertex2 = vbslq_f32(mask2, bbox_max, bbox_min);
        
        // Calculate dot products between planes and p-vertices
        float32x4_t mul_result1 = vmulq_f32(plane_vec1, p_vertex1);
        float32x4_t mul_result2 = vmulq_f32(plane_vec2, p_vertex2);
        
        // Horizontal adds to get dot products
        float32x2_t sum1 = vpadd_f32(vget_low_f32(mul_result1), vget_high_f32(mul_result1));
        sum1 = vpadd_f32(sum1, sum1);
        float dot_product1 = vget_lane_f32(sum1, 0);
        
        // Check first plane
        if (dot_product1 < 0.0f) {
            return false;
        }
        
        // Only check second plane if we have one in this batch
        if (i + 1 < 6) {
            float32x2_t sum2 = vpadd_f32(vget_low_f32(mul_result2), vget_high_f32(mul_result2));
            sum2 = vpadd_f32(sum2, sum2);
            float dot_product2 = vget_lane_f32(sum2, 0);
            
            if (dot_product2 < 0.0f) {
                return false;
            }
        }
    }
    
    // Add a fast-path early exit for small objects
    // Calculate the size of the bounding box
    float32x4_t size = vsubq_f32(bbox_max, bbox_min);
    float32x2_t size_low = vget_low_f32(size);
    float max_dimension = fmaxf(vget_lane_f32(size_low, 0), 
                               fmaxf(vget_lane_f32(size_low, 1), vget_lane_f32(vget_high_f32(size), 0)));
    
    // If the object is very small, it's likely not important for rendering
    // This helps with games that have many small decorative elements
    static const float SMALL_OBJECT_THRESHOLD = 0.01f;
    if (max_dimension < SMALL_OBJECT_THRESHOLD) {
        // For very small objects, apply stricter culling
        float32x4_t center = vmulq_n_f32(vaddq_f32(bbox_min, bbox_max), 0.5f);
        
        // Check if the center is far from the camera
        float32x2_t center_xy = vget_low_f32(center);
        float distance_sq = vget_lane_f32(center_xy, 0) * vget_lane_f32(center_xy, 0) + 
                           vget_lane_f32(center_xy, 1) * vget_lane_f32(center_xy, 1) + 
                           vget_lane_f32(vget_high_f32(center), 0) * vget_lane_f32(vget_high_f32(center), 0);
        
        // Small objects far away can be culled more aggressively
        static const float DISTANCE_THRESHOLD = 100.0f;
        if (distance_sq > DISTANCE_THRESHOLD * DISTANCE_THRESHOLD) {
            return false;
        }
    }
#else
    // For each plane, check if the bounding box is completely outside
    for (const auto& plane : frustum.planes) {
        // Find the p-vertex (furthest point in the normal direction)
        Common::Vec3<float> p_vertex;
        p_vertex.x = (plane.x > 0.0f) ? bbox.max.x : bbox.min.x;
        p_vertex.y = (plane.y > 0.0f) ? bbox.max.y : bbox.min.y;
        p_vertex.z = (plane.z > 0.0f) ? bbox.max.z : bbox.min.z;
        
        // Calculate dot product: dot(plane, p_vertex) + plane.w
        float dot_product = plane.x * p_vertex.x + 
                           plane.y * p_vertex.y + 
                           plane.z * p_vertex.z + 
                           plane.w;
        
        // If the p-vertex is outside the plane, the entire box is outside the frustum
        if (dot_product < 0.0f) {
            return false;
        }
    }
#endif
    
    return true;
}

bool GeometryCulling::IsPointVisible(const Common::Vec3<float>& point, 
                                     const CullingFrustumPlanes& frustum) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Test against each frustum plane using NEON
    for (const auto& plane : frustum.planes) {
        // Create a point vector with w=1 more efficiently
        float32x4_t point_vec;
        
        // Load XYZ components directly from memory when possible
        // This avoids multiple vset operations
        #if defined(__aarch64__)
        // On ARM64, we can use a more efficient approach
        point_vec = vdupq_n_f32(0.0f);
        // Use direct memory access if the point is properly aligned
        if (((uintptr_t)&point.x & 0xF) == 0) {
            // Load directly from memory if aligned
            point_vec = vld1q_f32(&point.x);
            // Set w component to 1.0f
            point_vec = vsetq_lane_f32(1.0f, point_vec, 3);
        } else {
            // Fall back to individual lane setting if not aligned
            point_vec = vsetq_lane_f32(point.x, point_vec, 0);
            point_vec = vsetq_lane_f32(point.y, point_vec, 1);
            point_vec = vsetq_lane_f32(point.z, point_vec, 2);
            point_vec = vsetq_lane_f32(1.0f, point_vec, 3);
        }
        #else
        // For older ARM NEON, use the original approach
        point_vec = vdupq_n_f32(0.0f);
        point_vec = vsetq_lane_f32(point.x, point_vec, 0);
        point_vec = vsetq_lane_f32(point.y, point_vec, 1);
        point_vec = vsetq_lane_f32(point.z, point_vec, 2);
        point_vec = vsetq_lane_f32(1.0f, point_vec, 3);
        #endif
        
        // Load plane coefficients
        float32x4_t plane_vec = vld1q_f32(&plane.x);
        
        // Calculate dot product using optimized NEON instructions
        float32x4_t mul_result = vmulq_f32(plane_vec, point_vec);
        
        // Sum the components to get the dot product
        float32x2_t sum = vpadd_f32(vget_low_f32(mul_result), vget_high_f32(mul_result));
        sum = vpadd_f32(sum, sum);
        float dot_product = vget_lane_f32(sum, 0);
        
        // If point is outside any plane, it's outside the frustum
        if (dot_product < 0.0f) {
            return false;
        }
    }
#else
    // Test against each frustum plane
    for (const auto& plane : frustum.planes) {
        // Calculate dot product: dot(plane.xyz, point) + plane.w
        float dot_product = plane.x * point.x + 
                           plane.y * point.y + 
                           plane.z * point.z + 
                           plane.w;
        
        // If point is outside any plane, it's outside the frustum
        if (dot_product < 0.0f) {
            return false;
        }
    }
#endif
    
    return true;
}

CullingFrustumPlanes GeometryCulling::CalculateFrustumPlanes(const Matrix4x4& view_projection) {
    CullingFrustumPlanes frustum;
    
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Extract rows from the view-projection matrix
    float32x4_t row0 = vld1q_f32(&view_projection.r[0].x);
    float32x4_t row1 = vld1q_f32(&view_projection.r[1].x);
    float32x4_t row2 = vld1q_f32(&view_projection.r[2].x);
    float32x4_t row3 = vld1q_f32(&view_projection.r[3].x);
    
    // Prefetch the memory locations where we'll store the planes
    // This helps the CPU pipeline by indicating our future memory access pattern
    __builtin_prefetch(&frustum.planes[0].x, 1); // 1 = for write
    __builtin_prefetch(&frustum.planes[3].x, 1);
    
    // Process planes in pairs to maximize NEON pipeline utilization
    // Left and right planes (row3 +/- row0)
    float32x4_t left_plane = vaddq_f32(row3, row0);
    float32x4_t right_plane = vsubq_f32(row3, row0);
    
    // Store left and right planes
    vst1q_f32(&frustum.planes[0].x, left_plane);
    vst1q_f32(&frustum.planes[1].x, right_plane);
    
    // Bottom and top planes (row3 +/- row1)
    float32x4_t bottom_plane = vaddq_f32(row3, row1);
    float32x4_t top_plane = vsubq_f32(row3, row1);
    
    // Store bottom and top planes
    vst1q_f32(&frustum.planes[2].x, bottom_plane);
    vst1q_f32(&frustum.planes[3].x, top_plane);
    
    // Near and far planes (row3 +/- row2)
    float32x4_t near_plane = vaddq_f32(row3, row2);
    float32x4_t far_plane = vsubq_f32(row3, row2);
    
    // Store near and far planes
    vst1q_f32(&frustum.planes[4].x, near_plane);
    vst1q_f32(&frustum.planes[5].x, far_plane);
    
    // Prefetch the normalization data we'll need soon
    // This helps reduce cache misses in the normalization loop
    for (int i = 0; i < 6; i++) {
        __builtin_prefetch(&frustum.planes[i].x, 1);
    }
#else
    // Left plane (row3 + row0)
    frustum.planes[0] = {view_projection.r[3].x + view_projection.r[0].x,
                        view_projection.r[3].y + view_projection.r[0].y,
                        view_projection.r[3].z + view_projection.r[0].z,
                        view_projection.r[3].w + view_projection.r[0].w};
    
    // Right plane (row3 - row0)
    frustum.planes[1] = {view_projection.r[3].x - view_projection.r[0].x,
                        view_projection.r[3].y - view_projection.r[0].y,
                        view_projection.r[3].z - view_projection.r[0].z,
                        view_projection.r[3].w - view_projection.r[0].w};
    
    // Bottom plane (row3 + row1)
    frustum.planes[2] = {view_projection.r[3].x + view_projection.r[1].x,
                        view_projection.r[3].y + view_projection.r[1].y,
                        view_projection.r[3].z + view_projection.r[1].z,
                        view_projection.r[3].w + view_projection.r[1].w};
    
    // Top plane (row3 - row1)
    frustum.planes[3] = {view_projection.r[3].x - view_projection.r[1].x,
                        view_projection.r[3].y - view_projection.r[1].y,
                        view_projection.r[3].z - view_projection.r[1].z,
                        view_projection.r[3].w - view_projection.r[1].w};
    
    // Near plane (row3 + row2)
    frustum.planes[4] = {view_projection.r[3].x + view_projection.r[2].x,
                        view_projection.r[3].y + view_projection.r[2].y,
                        view_projection.r[3].z + view_projection.r[2].z,
                        view_projection.r[3].w + view_projection.r[2].w};
    
    // Far plane (row3 - row2)
    frustum.planes[5] = {view_projection.r[3].x - view_projection.r[2].x,
                        view_projection.r[3].y - view_projection.r[2].y,
                        view_projection.r[3].z - view_projection.r[2].z,
                        view_projection.r[3].w - view_projection.r[2].w};
#endif
    
    // Normalize all planes
    for (auto& plane : frustum.planes) {
#if defined(__ARM_NEON) || defined(__aarch64__)
        // Load plane coefficients
        float32x4_t plane_vec = vld1q_f32(&plane.x);
        
        // Extract normal components (xyz) into a vector
        float32x4_t normal_vec = plane_vec;
        normal_vec = vsetq_lane_f32(0.0f, normal_vec, 3);  // Set w to 0 to exclude from length calculation
        
        // Calculate squared length using dot product with itself
        float32x4_t normal_squared = vmulq_f32(normal_vec, normal_vec);
        
        // Horizontal add to get sum of squares efficiently
        // Add pairs within each half of the vector
        float32x2_t sum_halves = vpadd_f32(vget_low_f32(normal_squared), vget_high_f32(normal_squared));
        // Add the two halves together
        float32x2_t sum = vpadd_f32(sum_halves, sum_halves);
        
        // Extract the squared length
        float length_squared = vget_lane_f32(sum, 0);
        
        // Calculate reciprocal square root (1/sqrt) - more efficient than division
        // Use vrsqrteq_f32 for an approximate reciprocal square root, then refine with Newton-Raphson
        float length_recip = vget_lane_f32(vrsqrte_f32(vdup_n_f32(length_squared)), 0);
        
        // One Newton-Raphson refinement step for better accuracy
        // new_est = est * (1.5f - 0.5f * est * est * x)
        length_recip = length_recip * (1.5f - 0.5f * length_squared * length_recip * length_recip);
        
        // Create vector with reciprocal length
        float32x4_t length_recip_vec = vdupq_n_f32(length_recip);
        
        // Multiply by reciprocal length to normalize (faster than division)
        float32x4_t normalized_plane = vmulq_f32(plane_vec, length_recip_vec);
        
        // Store normalized plane
        vst1q_f32(&plane.x, normalized_plane);
#else
        // Extract normal (xyz components)
        Common::Vec3<float> normal = {plane.x, plane.y, plane.z};
        
        // Calculate length squared
        float length_squared = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
        
        // Calculate reciprocal square root (1/sqrt) - more efficient than division
        float length_recip = 1.0f / sqrtf(length_squared);
        
        // Normalize the plane by multiplying by reciprocal length
        plane.x *= length_recip;
        plane.y *= length_recip;
        plane.z *= length_recip;
        plane.w *= length_recip;
#endif
    }
    
    return frustum;
}

bool GeometryCulling::IsTriangleFacingCamera(
    const Common::Vec3<float>& v0,
    const Common::Vec3<float>& v1,
    const Common::Vec3<float>& v2,
    const Common::Vec3<float>& view_position) {
    
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Load vertices into NEON registers
    float32x4_t v0_vec = vdupq_n_f32(0.0f);
    float32x4_t v1_vec = vdupq_n_f32(0.0f);
    float32x4_t v2_vec = vdupq_n_f32(0.0f);
    float32x4_t view_vec = vdupq_n_f32(0.0f);
    
    // Set XYZ components (leaving W as 0)
    v0_vec = vsetq_lane_f32(v0.x, v0_vec, 0);
    v0_vec = vsetq_lane_f32(v0.y, v0_vec, 1);
    v0_vec = vsetq_lane_f32(v0.z, v0_vec, 2);
    
    v1_vec = vsetq_lane_f32(v1.x, v1_vec, 0);
    v1_vec = vsetq_lane_f32(v1.y, v1_vec, 1);
    v1_vec = vsetq_lane_f32(v1.z, v1_vec, 2);
    
    v2_vec = vsetq_lane_f32(v2.x, v2_vec, 0);
    v2_vec = vsetq_lane_f32(v2.y, v2_vec, 1);
    v2_vec = vsetq_lane_f32(v2.z, v2_vec, 2);
    
    view_vec = vsetq_lane_f32(view_position.x, view_vec, 0);
    view_vec = vsetq_lane_f32(view_position.y, view_vec, 1);
    view_vec = vsetq_lane_f32(view_position.z, view_vec, 2);
    
    // Calculate edge vectors
    float32x4_t edge1 = vsubq_f32(v1_vec, v0_vec);
    float32x4_t edge2 = vsubq_f32(v2_vec, v0_vec);
    
    // Extract components for cross product
    float edge1_x = vgetq_lane_f32(edge1, 0);
    float edge1_y = vgetq_lane_f32(edge1, 1);
    float edge1_z = vgetq_lane_f32(edge1, 2);
    
    float edge2_x = vgetq_lane_f32(edge2, 0);
    float edge2_y = vgetq_lane_f32(edge2, 1);
    float edge2_z = vgetq_lane_f32(edge2, 2);
    
    // Calculate normal using cross product
    float32x4_t normal = vdupq_n_f32(0.0f);
    normal = vsetq_lane_f32(edge1_y * edge2_z - edge1_z * edge2_y, normal, 0); // normal.x
    normal = vsetq_lane_f32(edge1_z * edge2_x - edge1_x * edge2_z, normal, 1); // normal.y
    normal = vsetq_lane_f32(edge1_x * edge2_y - edge1_y * edge2_x, normal, 2); // normal.z
    
    // Calculate view direction (from triangle to camera)
    float32x4_t view_dir = vsubq_f32(view_vec, v0_vec);
    
    // Calculate dot product between normal and view direction using NEON
    float32x4_t mul_result = vmulq_f32(normal, view_dir);
    
    // Sum the first 3 components (x, y, z) to get the dot product
    float dot_product = vgetq_lane_f32(mul_result, 0) + 
                       vgetq_lane_f32(mul_result, 1) + 
                       vgetq_lane_f32(mul_result, 2);
#else
    // Calculate edge vectors
    Common::Vec3<float> edge1 = {
        v1.x - v0.x,
        v1.y - v0.y,
        v1.z - v0.z
    };
    
    Common::Vec3<float> edge2 = {
        v2.x - v0.x,
        v2.y - v0.y,
        v2.z - v0.z
    };
    
    // Calculate normal using cross product
    Common::Vec3<float> normal = {
        edge1.y * edge2.z - edge1.z * edge2.y,
        edge1.z * edge2.x - edge1.x * edge2.z,
        edge1.x * edge2.y - edge1.y * edge2.x
    };
    
    // Calculate view direction (from triangle to camera)
    Common::Vec3<float> view_dir = {
        view_position.x - v0.x,
        view_position.y - v0.y,
        view_position.z - v0.z
    };
    
    // Calculate dot product between normal and view direction
    float dot_product = normal.x * view_dir.x + normal.y * view_dir.y + normal.z * view_dir.z;
#endif
    
    // If dot product is positive, triangle is facing the camera
    return dot_product > 0.0f;
}

} // namespace Pica
