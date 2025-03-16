// Copyright 2025 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstddef>
#include <cstdint>

// Only use ARM NEON intrinsics on ARM64 platforms
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#include "common/memory_ops_neon.h"

namespace Pica {
namespace Swizzle {
namespace NEON {

/**
 * Optimized texture swizzling from linear layout to Morton order using NEON intrinsics
 * @param dest Destination buffer (Morton order)
 * @param source Source buffer (linear layout)
 * @param width Texture width
 * @param height Texture height
 * @param bytes_per_pixel Bytes per pixel (1, 2, 4, or 8)
 */
inline void SwizzleTexture(uint8_t* dest, const uint8_t* source, uint32_t width, uint32_t height, uint32_t bytes_per_pixel) {
    // Precompute lookup tables for X and Y coordinates
    // These tables map linear coordinates to their respective Morton bits
    uint32_t x_table[256];
    uint32_t y_table[256];
    
    for (uint32_t i = 0; i < 256; i++) {
        // Interleave bits for X coordinate
        uint32_t x = 0;
        for (uint32_t j = 0; j < 8; j++) {
            x |= ((i >> j) & 1) << (j * 2);
        }
        x_table[i] = x;
        
        // Interleave bits for Y coordinate
        uint32_t y = 0;
        for (uint32_t j = 0; j < 8; j++) {
            y |= ((i >> j) & 1) << (j * 2 + 1);
        }
        y_table[i] = y;
    }
    
    // Process pixels in batches for better cache utilization
    const uint32_t batch_size = 16;
    uint32_t morton_indices[batch_size];
    
    for (uint32_t y_batch = 0; y_batch < height; y_batch += batch_size) {
        for (uint32_t x_batch = 0; x_batch < width; x_batch += batch_size) {
            // Calculate Morton indices for this batch
            for (uint32_t y_offset = 0; y_offset < batch_size && y_batch + y_offset < height; y_offset++) {
                for (uint32_t x_offset = 0; x_offset < batch_size && x_batch + x_offset < width; x_offset++) {
                    uint32_t x = x_batch + x_offset;
                    uint32_t y = y_batch + y_offset;
                    
                    // Calculate Morton index using lookup tables
                    uint32_t morton = x_table[x & 0xFF] | y_table[y & 0xFF];
                    
                    // Handle larger coordinates
                    if (width > 256 || height > 256) {
                        morton |= (x_table[(x >> 8) & 0xFF] | y_table[(y >> 8) & 0xFF]) << 16;
                    }
                    
                    uint32_t batch_index = y_offset * batch_size + x_offset;
                    if (batch_index < batch_size) {
                        morton_indices[batch_index] = morton;
                    }
                }
            }
            
            // Copy pixels using the calculated Morton indices
            for (uint32_t i = 0; i < batch_size; i++) {
                uint32_t x_offset = i % batch_size;
                uint32_t y_offset = i / batch_size;
                
                if (x_batch + x_offset < width && y_batch + y_offset < height) {
                    uint32_t linear_index = (y_batch + y_offset) * width + (x_batch + x_offset);
                    uint32_t morton_index = morton_indices[i];
                    
                    // Copy pixel data
                    switch (bytes_per_pixel) {
                    case 1:
                        dest[morton_index] = source[linear_index];
                        break;
                    case 2:
                        *reinterpret_cast<uint16_t*>(dest + morton_index * 2) = 
                            *reinterpret_cast<const uint16_t*>(source + linear_index * 2);
                        break;
                    case 4:
                        *reinterpret_cast<uint32_t*>(dest + morton_index * 4) = 
                            *reinterpret_cast<const uint32_t*>(source + linear_index * 4);
                        break;
                    case 8:
                        // Use NEON for 8-byte copies
                        uint8x8_t pixel = vld1_u8(source + linear_index * 8);
                        vst1_u8(dest + morton_index * 8, pixel);
                        break;
                    }
                }
            }
        }
    }
}

/**
 * Optimized texture unswizzling from Morton order to linear layout using NEON intrinsics
 * @param dest Destination buffer (linear layout)
 * @param source Source buffer (Morton order)
 * @param width Texture width
 * @param height Texture height
 * @param bytes_per_pixel Bytes per pixel (1, 2, 4, or 8)
 */
inline void UnswizzleTexture(uint8_t* dest, const uint8_t* source, uint32_t width, uint32_t height, uint32_t bytes_per_pixel) {
    // Create lookup tables for fast Morton decode
    uint32_t morton_x[65536];
    uint32_t morton_y[65536];
    
    for (uint32_t i = 0; i < 65536; i++) {
        // Extract X coordinate (even bits)
        uint32_t x = 0;
        for (uint32_t j = 0; j < 16; j++) {
            x |= ((i >> (j * 2)) & 1) << j;
        }
        morton_x[i] = x;
        
        // Extract Y coordinate (odd bits)
        uint32_t y = 0;
        for (uint32_t j = 0; j < 16; j++) {
            y |= ((i >> (j * 2 + 1)) & 1) << j;
        }
        morton_y[i] = y;
    }
    
    // Process pixels in scanlines for better cache utilization
    for (uint32_t y = 0; y < height; y++) {
        // Use NEON to process multiple pixels at once when possible
        uint32_t x = 0;
        
        // Process 8 pixels at a time for 4 bytes per pixel
        if (bytes_per_pixel == 4 && width >= 8) {
            for (; x + 7 < width; x += 8) {
                uint32_t indices[8];
                
                // Calculate Morton indices
                for (uint32_t i = 0; i < 8; i++) {
                    uint32_t morton = 0;
                    uint32_t x_coord = x + i;
                    
                    // Calculate Morton index
                    morton |= morton_x[x_coord & 0xFFFF] | (morton_y[y & 0xFFFF] << 1);
                    
                    indices[i] = morton;
                }
                
                // Load pixels from Morton order and store in linear order
                uint32x4_t pixels_low = vld1q_u32(reinterpret_cast<const uint32_t*>(source + indices[0] * 4));
                uint32x4_t pixels_high = vld1q_u32(reinterpret_cast<const uint32_t*>(source + indices[4] * 4));
                
                // Store in linear order
                vst1q_u32(reinterpret_cast<uint32_t*>(dest + (y * width + x) * 4), pixels_low);
                vst1q_u32(reinterpret_cast<uint32_t*>(dest + (y * width + x + 4) * 4), pixels_high);
            }
        }
        
        // Handle remaining pixels
        for (; x < width; x++) {
            uint32_t morton = 0;
            
            // Calculate Morton index
            morton |= morton_x[x & 0xFFFF] | (morton_y[y & 0xFFFF] << 1);
            
            // Copy pixel data
            switch (bytes_per_pixel) {
            case 1:
                dest[y * width + x] = source[morton];
                break;
            case 2:
                *reinterpret_cast<uint16_t*>(dest + (y * width + x) * 2) = 
                    *reinterpret_cast<const uint16_t*>(source + morton * 2);
                break;
            case 4:
                *reinterpret_cast<uint32_t*>(dest + (y * width + x) * 4) = 
                    *reinterpret_cast<const uint32_t*>(source + morton * 4);
                break;
            case 8:
                // Use NEON for 8-byte copies
                uint8x8_t pixel = vld1_u8(source + morton * 8);
                vst1_u8(dest + (y * width + x) * 8, pixel);
                break;
            }
        }
    }
}

/**
 * Optimized texture swizzling for ETC1 compressed textures using NEON intrinsics
 * @param dest Destination buffer (Morton order)
 * @param source Source buffer (linear layout)
 * @param width Texture width (in blocks, width/4)
 * @param height Texture height (in blocks, height/4)
 */
inline void SwizzleETC1Texture(uint8_t* dest, const uint8_t* source, uint32_t width, uint32_t height) {
    // ETC1 uses 8 bytes per 4x4 pixel block
    SwizzleTexture(dest, source, width, height, 8);
}

/**
 * Optimized texture unswizzling for ETC1 compressed textures using NEON intrinsics
 * @param dest Destination buffer (linear layout)
 * @param source Source buffer (Morton order)
 * @param width Texture width (in blocks, width/4)
 * @param height Texture height (in blocks, height/4)
 */
inline void UnswizzleETC1Texture(uint8_t* dest, const uint8_t* source, uint32_t width, uint32_t height) {
    // ETC1 uses 8 bytes per 4x4 pixel block
    UnswizzleTexture(dest, source, width, height, 8);
}

/**
 * Optimized framebuffer swizzling from linear layout to tiled layout using NEON intrinsics
 * @param dest Destination buffer (tiled layout)
 * @param source Source buffer (linear layout)
 * @param width Framebuffer width
 * @param height Framebuffer height
 * @param bytes_per_pixel Bytes per pixel (typically 4 for RGBA8)
 */
inline void SwizzleFramebuffer(uint8_t* dest, const uint8_t* source, uint32_t width, uint32_t height, uint32_t bytes_per_pixel) {
    // 3DS uses 8x8 tiles for framebuffers
    const uint32_t tile_size = 8;
    const uint32_t tile_bytes = tile_size * tile_size * bytes_per_pixel;
    
    // Process each tile
    for (uint32_t y_tile = 0; y_tile < height; y_tile += tile_size) {
        for (uint32_t x_tile = 0; x_tile < width; x_tile += tile_size) {
            // Calculate actual tile dimensions (handle edge cases)
            uint32_t tile_width = std::min(tile_size, width - x_tile);
            uint32_t tile_height = std::min(tile_size, height - y_tile);
            
            // Process each pixel in the tile
            for (uint32_t y = 0; y < tile_height; y++) {
                // Use NEON to process entire rows at once
                uint32_t x = 0;
                
                // Process 8 pixels (32 bytes) at a time for RGBA8
                if (bytes_per_pixel == 4 && tile_width == 8) {
                    // Source: linear layout
                    const uint8_t* src_row = source + ((y_tile + y) * width + x_tile) * bytes_per_pixel;
                    
                    // Destination: tiled layout
                    uint8_t* dst_row = dest + ((y_tile / tile_size) * (width / tile_size) + (x_tile / tile_size)) * tile_bytes +
                                      y * tile_size * bytes_per_pixel;
                    
                    // Load 8 pixels (32 bytes) from linear layout
                    uint8x16_t pixels_low = vld1q_u8(src_row);
                    uint8x16_t pixels_high = vld1q_u8(src_row + 16);
                    
                    // Store in tiled layout
                    vst1q_u8(dst_row, pixels_low);
                    vst1q_u8(dst_row + 16, pixels_high);
                    
                    x = 8; // All pixels in this row are processed
                }
                
                // Handle remaining pixels
                for (; x < tile_width; x++) {
                    // Calculate source and destination indices
                    uint32_t src_index = ((y_tile + y) * width + (x_tile + x)) * bytes_per_pixel;
                    uint32_t dst_index = ((y_tile / tile_size) * (width / tile_size) + (x_tile / tile_size)) * tile_bytes +
                                        (y * tile_size + x) * bytes_per_pixel;
                    
                    // Copy pixel data
                    switch (bytes_per_pixel) {
                    case 1:
                        dest[dst_index] = source[src_index];
                        break;
                    case 2:
                        *reinterpret_cast<uint16_t*>(dest + dst_index) = 
                            *reinterpret_cast<const uint16_t*>(source + src_index);
                        break;
                    case 4:
                        *reinterpret_cast<uint32_t*>(dest + dst_index) = 
                            *reinterpret_cast<const uint32_t*>(source + src_index);
                        break;
                    }
                }
            }
        }
    }
}

/**
 * Optimized framebuffer unswizzling from tiled layout to linear layout using NEON intrinsics
 * @param dest Destination buffer (linear layout)
 * @param source Source buffer (tiled layout)
 * @param width Framebuffer width
 * @param height Framebuffer height
 * @param bytes_per_pixel Bytes per pixel (typically 4 for RGBA8)
 */
inline void UnswizzleFramebuffer(uint8_t* dest, const uint8_t* source, uint32_t width, uint32_t height, uint32_t bytes_per_pixel) {
    // 3DS uses 8x8 tiles for framebuffers
    const uint32_t tile_size = 8;
    const uint32_t tile_bytes = tile_size * tile_size * bytes_per_pixel;
    
    // Process each tile
    for (uint32_t y_tile = 0; y_tile < height; y_tile += tile_size) {
        for (uint32_t x_tile = 0; x_tile < width; x_tile += tile_size) {
            // Calculate actual tile dimensions (handle edge cases)
            uint32_t tile_width = std::min(tile_size, width - x_tile);
            uint32_t tile_height = std::min(tile_size, height - y_tile);
            
            // Process each pixel in the tile
            for (uint32_t y = 0; y < tile_height; y++) {
                // Use NEON to process entire rows at once
                uint32_t x = 0;
                
                // Process 8 pixels (32 bytes) at a time for RGBA8
                if (bytes_per_pixel == 4 && tile_width == 8) {
                    // Source: tiled layout
                    const uint8_t* src_row = source + ((y_tile / tile_size) * (width / tile_size) + (x_tile / tile_size)) * tile_bytes +
                                           y * tile_size * bytes_per_pixel;
                    
                    // Destination: linear layout
                    uint8_t* dst_row = dest + ((y_tile + y) * width + x_tile) * bytes_per_pixel;
                    
                    // Load 8 pixels (32 bytes) from tiled layout
                    uint8x16_t pixels_low = vld1q_u8(src_row);
                    uint8x16_t pixels_high = vld1q_u8(src_row + 16);
                    
                    // Store in linear layout
                    vst1q_u8(dst_row, pixels_low);
                    vst1q_u8(dst_row + 16, pixels_high);
                    
                    x = 8; // All pixels in this row are processed
                }
                
                // Handle remaining pixels
                for (; x < tile_width; x++) {
                    // Calculate source and destination indices
                    uint32_t src_index = ((y_tile / tile_size) * (width / tile_size) + (x_tile / tile_size)) * tile_bytes +
                                       (y * tile_size + x) * bytes_per_pixel;
                    uint32_t dst_index = ((y_tile + y) * width + (x_tile + x)) * bytes_per_pixel;
                    
                    // Copy pixel data
                    switch (bytes_per_pixel) {
                    case 1:
                        dest[dst_index] = source[src_index];
                        break;
                    case 2:
                        *reinterpret_cast<uint16_t*>(dest + dst_index) = 
                            *reinterpret_cast<const uint16_t*>(source + src_index);
                        break;
                    case 4:
                        *reinterpret_cast<uint32_t*>(dest + dst_index) = 
                            *reinterpret_cast<const uint32_t*>(source + src_index);
                        break;
                    }
                }
            }
        }
    }
}

/**
 * Optimized depth buffer swizzling from linear layout to tiled layout using NEON intrinsics
 * @param dest Destination buffer (tiled layout)
 * @param source Source buffer (linear layout)
 * @param width Depth buffer width
 * @param height Depth buffer height
 */
inline void SwizzleDepthBuffer(uint8_t* dest, const uint8_t* source, uint32_t width, uint32_t height) {
    // 3DS uses 8x8 tiles for depth buffers with 16-bit depth values
    SwizzleFramebuffer(dest, source, width, height, 2);
}

/**
 * Optimized depth buffer unswizzling from tiled layout to linear layout using NEON intrinsics
 * @param dest Destination buffer (linear layout)
 * @param source Source buffer (tiled layout)
 * @param width Depth buffer width
 * @param height Depth buffer height
 */
inline void UnswizzleDepthBuffer(uint8_t* dest, const uint8_t* source, uint32_t width, uint32_t height) {
    // 3DS uses 8x8 tiles for depth buffers with 16-bit depth values
    UnswizzleFramebuffer(dest, source, width, height, 2);
}

} // namespace NEON
} // namespace Swizzle
} // namespace Pica

#endif // defined(__ARM_NEON) || defined(__aarch64__)
