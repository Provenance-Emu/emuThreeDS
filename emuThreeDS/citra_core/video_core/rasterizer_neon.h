// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>

// Only use ARM NEON intrinsics on ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#include "common/vector_math.h"
#include "common/matrix_math_neon.h"
#include "video_core/pica_types.h"
#include "video_core/shader_processing_neon.h"

namespace Pica {
namespace Rasterizer {
namespace NEON {

/**
 * Optimized edge function calculation using NEON intrinsics
 * @param vertex0 First vertex of the edge
 * @param vertex1 Second vertex of the edge
 * @param point Point to test
 * @return Edge function value
 */
inline float EdgeFunction(const float32x2_t& vertex0, const float32x2_t& vertex1, const float32x2_t& point) {
    float32x2_t edge = vsub_f32(vertex1, vertex0);
    float32x2_t to_point = vsub_f32(point, vertex0);
    
    // Calculate cross product: edge.x * to_point.y - edge.y * to_point.x
    float edge_x = vget_lane_f32(edge, 0);
    float edge_y = vget_lane_f32(edge, 1);
    float to_point_x = vget_lane_f32(to_point, 0);
    float to_point_y = vget_lane_f32(to_point, 1);
    
    return edge_x * to_point_y - edge_y * to_point_x;
}

/**
 * Optimized batch edge function calculation using NEON intrinsics
 * @param edge_values Output edge function values
 * @param x_coords X coordinates of the points to test
 * @param y_coords Y coordinates of the points to test
 * @param v0x X coordinate of the first vertex
 * @param v0y Y coordinate of the first vertex
 * @param v1x X coordinate of the second vertex
 * @param v1y Y coordinate of the second vertex
 * @param count Number of points to test
 */
inline void BatchEdgeFunction(float* edge_values, const float* x_coords, const float* y_coords,
                             float v0x, float v0y, float v1x, float v1y, size_t count) {
    // Create NEON registers for the edge vertices
    float32x2_t vertex0 = {v0x, v0y};
    float32x2_t vertex1 = {v1x, v1y};
    float32x2_t edge = vsub_f32(vertex1, vertex0);
    
    // Process points in batches of 4
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        // Load 4 points
        float32x4_t x_batch = vld1q_f32(&x_coords[i]);
        float32x4_t y_batch = vld1q_f32(&y_coords[i]);
        
        // Calculate edge function for each point
        float edge_x = vget_lane_f32(edge, 0);
        float edge_y = vget_lane_f32(edge, 1);
        
        // Extract individual values from x_batch and y_batch
        float x0 = vgetq_lane_f32(x_batch, 0);
        float y0 = vgetq_lane_f32(y_batch, 0);
        float x1 = vgetq_lane_f32(x_batch, 1);
        float y1 = vgetq_lane_f32(y_batch, 1);
        float x2 = vgetq_lane_f32(x_batch, 2);
        float y2 = vgetq_lane_f32(y_batch, 2);
        float x3 = vgetq_lane_f32(x_batch, 3);
        float y3 = vgetq_lane_f32(y_batch, 3);
        
        // Calculate edge function for each point individually
        float32x2_t point0 = {x0, y0};
        float32x2_t to_point0 = vsub_f32(point0, vertex0);
        float to_point0_x = vget_lane_f32(to_point0, 0);
        float to_point0_y = vget_lane_f32(to_point0, 1);
        float result0 = edge_x * to_point0_y - edge_y * to_point0_x;
        
        float32x2_t point1 = {x1, y1};
        float32x2_t to_point1 = vsub_f32(point1, vertex0);
        float to_point1_x = vget_lane_f32(to_point1, 0);
        float to_point1_y = vget_lane_f32(to_point1, 1);
        float result1 = edge_x * to_point1_y - edge_y * to_point1_x;
        
        float32x2_t point2 = {x2, y2};
        float32x2_t to_point2 = vsub_f32(point2, vertex0);
        float to_point2_x = vget_lane_f32(to_point2, 0);
        float to_point2_y = vget_lane_f32(to_point2, 1);
        float result2 = edge_x * to_point2_y - edge_y * to_point2_x;
        
        float32x2_t point3 = {x3, y3};
        float32x2_t to_point3 = vsub_f32(point3, vertex0);
        float to_point3_x = vget_lane_f32(to_point3, 0);
        float to_point3_y = vget_lane_f32(to_point3, 1);
        float result3 = edge_x * to_point3_y - edge_y * to_point3_x;
        
        // Create results vector with constant indices
        float32x4_t results = vdupq_n_f32(0.0f);
        results = vsetq_lane_f32(result0, results, 0);
        results = vsetq_lane_f32(result1, results, 1);
        results = vsetq_lane_f32(result2, results, 2);
        results = vsetq_lane_f32(result3, results, 3);
        
        // Store results
        vst1q_f32(&edge_values[i], results);
    }
    
    // Handle remaining points
    for (; i < count; i++) {
        float32x2_t point = {x_coords[i], y_coords[i]};
        edge_values[i] = EdgeFunction(vertex0, vertex1, point);
    }
}

/**
 * Optimized triangle rasterization using NEON intrinsics
 * @param framebuffer Output framebuffer
 * @param depth_buffer Output depth buffer
 * @param v0 First vertex
 * @param v1 Second vertex
 * @param v2 Third vertex
 * @param attrs0 Attributes of the first vertex
 * @param attrs1 Attributes of the second vertex
 * @param attrs2 Attributes of the third vertex
 * @param min_x Minimum X coordinate of the bounding box
 * @param min_y Minimum Y coordinate of the bounding box
 * @param max_x Maximum X coordinate of the bounding box
 * @param max_y Maximum Y coordinate of the bounding box
 * @param fb_width Framebuffer width
 * @param fb_height Framebuffer height
 */
template <typename AttributeType>
inline void RasterizeTriangle(Common::Vec4<u8>* framebuffer, float* depth_buffer,
                            const Common::Vec4<float>& v0, const Common::Vec4<float>& v1, const Common::Vec4<float>& v2,
                            const AttributeType& attrs0, const AttributeType& attrs1, const AttributeType& attrs2,
                            int min_x, int min_y, int max_x, int max_y, int fb_width, int fb_height) {
    // Calculate triangle area
    float32x2_t vertex0 = {v0.x / v0.w, v0.y / v0.w};
    float32x2_t vertex1 = {v1.x / v1.w, v1.y / v1.w};
    float32x2_t vertex2 = {v2.x / v2.w, v2.y / v2.w};
    
    float area = EdgeFunction(vertex0, vertex1, vertex2);
    if (std::abs(area) < 0.00001f) {
        return; // Degenerate triangle
    }
    
    float inv_area = 1.0f / area;
    
    // Prepare depth values
    float32x4_t depth_values = {v0.z / v0.w, v1.z / v1.w, v2.z / v2.w, 0.0f};
    
    // Rasterize the triangle
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            // Create point
            float32x2_t point = {static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
            
            // Calculate barycentric coordinates
            float e0 = EdgeFunction(vertex1, vertex2, point);
            float e1 = EdgeFunction(vertex2, vertex0, point);
            float e2 = EdgeFunction(vertex0, vertex1, point);
            
            // Check if the point is inside the triangle
            if (e0 >= 0.0f && e1 >= 0.0f && e2 >= 0.0f) {
                // Calculate barycentric weights
                float32x4_t weights = {e0 * inv_area, e1 * inv_area, e2 * inv_area, 0.0f};
                
                // Interpolate depth
                float32x2_t depth_terms = {
                    weights[0] * depth_values[0],
                    weights[1] * depth_values[1]
                };
                float depth = depth_terms[0] + depth_terms[1] + weights[2] * depth_values[2];
                
                // Depth test
                int fb_index = y * fb_width + x;
                if (depth < depth_buffer[fb_index]) {
                    // Update depth buffer
                    depth_buffer[fb_index] = depth;
                    
                    // Interpolate attributes
                    AttributeType interpolated_attrs;
                    Common::Vec3<float> bary_weights = {weights[0], weights[1], weights[2]};
                    
                    // Use the InterpolateAttribute function from shader_processing_neon.h
                    Pica::Shader::NEON::InterpolateAttribute(interpolated_attrs, attrs0, attrs1, attrs2, bary_weights);
                    
                    // Convert attribute to color
                    Common::Vec4<u8> color;
                    if constexpr (std::is_same_v<AttributeType, Common::Vec4<float>>) {
                        // Assume attribute is already a color in float format
                        color.r() = static_cast<u8>(std::clamp(interpolated_attrs.x * 255.0f, 0.0f, 255.0f));
                        color.g() = static_cast<u8>(std::clamp(interpolated_attrs.y * 255.0f, 0.0f, 255.0f));
                        color.b() = static_cast<u8>(std::clamp(interpolated_attrs.z * 255.0f, 0.0f, 255.0f));
                        color.a() = static_cast<u8>(std::clamp(interpolated_attrs.w * 255.0f, 0.0f, 255.0f));
                    } else {
                        // Default to white for other attribute types
                        color = {255, 255, 255, 255};
                    }
                    
                    // Write to framebuffer
                    framebuffer[fb_index] = color;
                }
            }
        }
    }
}

/**
 * Optimized triangle batch rasterization using NEON intrinsics
 * @param framebuffer Output framebuffer
 * @param depth_buffer Output depth buffer
 * @param vertices Array of vertices
 * @param attributes Array of vertex attributes
 * @param indices Array of vertex indices
 * @param index_count Number of indices (must be a multiple of 3)
 * @param fb_width Framebuffer width
 * @param fb_height Framebuffer height
 */
template <typename AttributeType>
inline void RasterizeTriangleBatch(Common::Vec4<u8>* framebuffer, float* depth_buffer,
                                 const Common::Vec4<float>* vertices,
                                 const AttributeType* attributes,
                                 const uint16_t* indices, size_t index_count,
                                 int fb_width, int fb_height) {
    // Process triangles
    for (size_t i = 0; i < index_count; i += 3) {
        // Get vertex indices
        uint16_t idx0 = indices[i];
        uint16_t idx1 = indices[i + 1];
        uint16_t idx2 = indices[i + 2];
        
        // Get vertices
        const Common::Vec4<float>& v0 = vertices[idx0];
        const Common::Vec4<float>& v1 = vertices[idx1];
        const Common::Vec4<float>& v2 = vertices[idx2];
        
        // Skip triangles with vertices outside the view frustum
        if (v0.w <= 0.0f || v1.w <= 0.0f || v2.w <= 0.0f) {
            continue;
        }
        
        // Calculate screen coordinates
        float32x2_t screen0 = {
            (v0.x / v0.w * 0.5f + 0.5f) * fb_width,
            (v0.y / v0.w * 0.5f + 0.5f) * fb_height
        };
        float32x2_t screen1 = {
            (v1.x / v1.w * 0.5f + 0.5f) * fb_width,
            (v1.y / v1.w * 0.5f + 0.5f) * fb_height
        };
        float32x2_t screen2 = {
            (v2.x / v2.w * 0.5f + 0.5f) * fb_width,
            (v2.y / v2.w * 0.5f + 0.5f) * fb_height
        };
        
        // Calculate bounding box
        int min_x = std::max(0, static_cast<int>(std::min({vget_lane_f32(screen0, 0), vget_lane_f32(screen1, 0), vget_lane_f32(screen2, 0)})));
        int min_y = std::max(0, static_cast<int>(std::min({vget_lane_f32(screen0, 1), vget_lane_f32(screen1, 1), vget_lane_f32(screen2, 1)})));
        int max_x = std::min(fb_width - 1, static_cast<int>(std::max({vget_lane_f32(screen0, 0), vget_lane_f32(screen1, 0), vget_lane_f32(screen2, 0)})));
        int max_y = std::min(fb_height - 1, static_cast<int>(std::max({vget_lane_f32(screen0, 1), vget_lane_f32(screen1, 1), vget_lane_f32(screen2, 1)})));
        
        // Get attributes
        const AttributeType& attrs0 = attributes[idx0];
        const AttributeType& attrs1 = attributes[idx1];
        const AttributeType& attrs2 = attributes[idx2];
        
        // Rasterize the triangle
        RasterizeTriangle(framebuffer, depth_buffer, v0, v1, v2, attrs0, attrs1, attrs2,
                         min_x, min_y, max_x, max_y, fb_width, fb_height);
    }
}

/**
 * Optimized tile-based triangle rasterization using NEON intrinsics
 * @param framebuffer Output framebuffer
 * @param depth_buffer Output depth buffer
 * @param v0 First vertex
 * @param v1 Second vertex
 * @param v2 Third vertex
 * @param attrs0 Attributes of the first vertex
 * @param attrs1 Attributes of the second vertex
 * @param attrs2 Attributes of the third vertex
 * @param min_x Minimum X coordinate of the bounding box
 * @param min_y Minimum Y coordinate of the bounding box
 * @param max_x Maximum X coordinate of the bounding box
 * @param max_y Maximum Y coordinate of the bounding box
 * @param fb_width Framebuffer width
 * @param fb_height Framebuffer height
 * @param tile_size Size of the tiles (e.g., 8x8 pixels)
 */
template <typename AttributeType>
inline void RasterizeTriangleTiled(Common::Vec4<u8>* framebuffer, float* depth_buffer,
                                 const Common::Vec4<float>& v0, const Common::Vec4<float>& v1, const Common::Vec4<float>& v2,
                                 const AttributeType& attrs0, const AttributeType& attrs1, const AttributeType& attrs2,
                                 int min_x, int min_y, int max_x, int max_y, int fb_width, int fb_height, int tile_size) {
    // Calculate triangle area
    float32x2_t vertex0 = {v0.x / v0.w, v0.y / v0.w};
    float32x2_t vertex1 = {v1.x / v1.w, v1.y / v1.w};
    float32x2_t vertex2 = {v2.x / v2.w, v2.y / v2.w};
    
    float area = EdgeFunction(vertex0, vertex1, vertex2);
    if (std::abs(area) < 0.00001f) {
        return; // Degenerate triangle
    }
    
    float inv_area = 1.0f / area;
    
    // Prepare depth values
    float32x4_t depth_values = {v0.z / v0.w, v1.z / v1.w, v2.z / v2.w, 0.0f};
    
    // Align bounding box to tile grid
    int min_tile_x = min_x / tile_size;
    int min_tile_y = min_y / tile_size;
    int max_tile_x = (max_x + tile_size - 1) / tile_size;
    int max_tile_y = (max_y + tile_size - 1) / tile_size;
    
    // Process tiles
    for (int tile_y = min_tile_y; tile_y <= max_tile_y; tile_y++) {
        for (int tile_x = min_tile_x; tile_x <= max_tile_x; tile_x++) {
            // Calculate tile bounds
            int tile_min_x = std::max(min_x, tile_x * tile_size);
            int tile_min_y = std::max(min_y, tile_y * tile_size);
            int tile_max_x = std::min(max_x, (tile_x + 1) * tile_size - 1);
            int tile_max_y = std::min(max_y, (tile_y + 1) * tile_size - 1);
            
            // Process pixels within the tile
            for (int y = tile_min_y; y <= tile_max_y; y++) {
                for (int x = tile_min_x; x <= tile_max_x; x++) {
                    // Create point
                    float32x2_t point = {static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
                    
                    // Calculate barycentric coordinates
                    float e0 = EdgeFunction(vertex1, vertex2, point);
                    float e1 = EdgeFunction(vertex2, vertex0, point);
                    float e2 = EdgeFunction(vertex0, vertex1, point);
                    
                    // Check if the point is inside the triangle
                    if (e0 >= 0.0f && e1 >= 0.0f && e2 >= 0.0f) {
                        // Calculate barycentric weights
                        float32x4_t weights = {e0 * inv_area, e1 * inv_area, e2 * inv_area, 0.0f};
                        
                        // Interpolate depth
                        float32x2_t depth_terms = {
                            weights[0] * depth_values[0],
                            weights[1] * depth_values[1]
                        };
                        float depth = depth_terms[0] + depth_terms[1] + weights[2] * depth_values[2];
                        
                        // Depth test
                        int fb_index = y * fb_width + x;
                        if (depth < depth_buffer[fb_index]) {
                            // Update depth buffer
                            depth_buffer[fb_index] = depth;
                            
                            // Interpolate attributes
                            AttributeType interpolated_attrs;
                            Common::Vec3<float> bary_weights = {weights[0], weights[1], weights[2]};
                            
                            // Use the InterpolateAttribute function from shader_processing_neon.h
                            Pica::Shader::NEON::InterpolateAttribute(interpolated_attrs, attrs0, attrs1, attrs2, bary_weights);
                            
                            // Convert attribute to color
                            Common::Vec4<u8> color;
                            if constexpr (std::is_same_v<AttributeType, Common::Vec4<float>>) {
                                // Assume attribute is already a color in float format
                                color.r() = static_cast<u8>(std::clamp(interpolated_attrs.x * 255.0f, 0.0f, 255.0f));
                                color.g() = static_cast<u8>(std::clamp(interpolated_attrs.y * 255.0f, 0.0f, 255.0f));
                                color.b() = static_cast<u8>(std::clamp(interpolated_attrs.z * 255.0f, 0.0f, 255.0f));
                                color.a() = static_cast<u8>(std::clamp(interpolated_attrs.w * 255.0f, 0.0f, 255.0f));
                            } else {
                                // Default to white for other attribute types
                                color = {255, 255, 255, 255};
                            }
                            
                            // Write to framebuffer
                            framebuffer[fb_index] = color;
                        }
                    }
                }
            }
        }
    }
}

} // namespace NEON
} // namespace Rasterizer
} // namespace Pica

#endif // defined(__ARM_NEON) || defined(__aarch64__)
